# SPDX-License-Identifier: Apache-2.0
"""Compiled MTP verify forward (lever E8).

The (K+1)-token verify forward dominates the CPU side of a spec-decode
round: ~4.2 ms of per-round Python graph construction (measured via
``RAPID_MLX_MTP_TIMING=1`` at K=2 on Qwen3.6-35B-A3B) against ~16.7 ms
of GPU work. ``mx.compile`` removes the rebuild by replaying a recorded
graph, but the stock forward is not compilable as-is:

* ``KVCache.offset`` is a Python int — baked into a trace as a constant,
  so a replay would reuse stale RoPE/mask positions.
* ``KVCache.update_and_fetch`` returns buffers sliced to ``offset+S`` —
  the shape changes every round, forcing a retrace per round.
* The patched ``GatedDeltaNet.__call__`` (cache_patch.py) writes
  ``cache.rollback_state`` and ``cache[0]/cache[1]`` mid-forward —
  side effects a compiled replay would silently skip.

The fix here is the MTPLX ``CompiledVerifyBank`` design (see
``mtplx/graphbank.py``), reduced to the stock-cache case:

* **Pure function**: ``verify_step(tokens, *state_in) -> (logits,
  hidden, *state_out)``. Every cache leaf enters as an explicit input
  and leaves as an explicit output, including the GDN rollback
  snapshots the chunk-split produces at each accept boundary.
* **Shadow-cache firewall**: the traced body re-seeds a bank-owned
  shadow cache list from the explicit inputs BEFORE any read, then runs
  the existing (side-effecting) model forward against the shadows.
  Tracers never escape into the real cache; the dispatcher
  mirror-commits materialized outputs back into the stock entries.
* **Tensor offsets + fixed shapes**: full-attention runs over the FULL
  256-step-aligned KV buffer with a boolean mask derived from an
  ``mx.array`` offset, so leaf shapes only change when a buffer grows.
  Measured on this machine (probe_e8, mlx 0.31.2, real 8-bit weights):
  full-buffer+mask attention, array-offset RoPE, and ``mx.compile`` of
  both layer types are all bit-identical to the eager path
  (max|diff| = 0.0), which is what keeps the greedy fingerprint gate
  intact.

Feature is env-gated and default OFF: ``RAPID_MLX_MTP_COMPILED_VERIFY=1``.
Any ineligible cache container, capacity overflow mid-growth, or
exception falls back to the eager forward for that call; three
consecutive failures disable the bank for the process.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_KIND_FA = "fa"
_KIND_GDN = "gdn"


def compiled_verify_enabled() -> bool:
    """True when RAPID_MLX_MTP_COMPILED_VERIFY requests the compiled path."""
    raw = os.environ.get("RAPID_MLX_MTP_COMPILED_VERIFY", "").strip().lower()
    return raw in ("1", "true", "on")


def _growth_reserve() -> int:
    """Extra KV headroom (tokens) granted when a buffer must grow.

    Sized so a typical generation completes inside one leaf-shape class:
    every capacity change forces ``mx.compile`` to retrace internally,
    and a mid-generation retrace is a decode stall. 256-token stock
    growth would retrace ~4x per 1k tokens; a 512-token reserve cuts
    that to ~1.3x. Env-tunable for A/B.
    """
    raw = os.environ.get("RAPID_MLX_MTP_COMPILED_VERIFY_RESERVE", "512").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 512


class _ShadowKVCache:
    """Fixed-capacity KV adapter with an ``mx.array`` offset.

    Stands in for a stock ``KVCache`` inside the traced verify body only.
    ``update_and_fetch`` writes the new rows via functional
    ``mx.slice_update`` at the tensor offset and returns the FULL
    buffers; ``make_mask`` masks out everything past the window, so the
    attention shapes are constant for a given buffer capacity.
    Bit-exactness of full-buffer+bool-mask vs sliced+"causal" was
    measured 0.0 across offsets 13..4093 (probe_e8, 2026-07-10).
    """

    def __init__(self) -> None:
        self.keys = None
        self.values = None
        self.offset = None  # mx.array, int32 scalar
        self.capacity = 0

    def seed(self, keys, values, offset) -> None:
        self.keys = keys
        self.values = values
        self.offset = offset
        self.capacity = int(keys.shape[2])

    def update_and_fetch(self, keys, values):
        steps = int(keys.shape[2])
        self.keys = mx.slice_update(self.keys, keys, self.offset, axes=(2,))
        self.values = mx.slice_update(self.values, values, self.offset, axes=(2,))
        self.offset = self.offset + steps
        return self.keys, self.values

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        del return_array
        rinds = mx.arange(self.capacity)
        linds = self.offset + mx.arange(N)
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + window_size)
        return mask


class CompiledVerify:
    """Process-lived bank of compiled verify callables for one model.

    Compiled entries are keyed by ``(S, n_confirmed)`` — ``mx.compile``
    retraces internally when input leaf shapes (KV capacities) change,
    and the traced body reads the bank's LIVE shadow list and re-seeds
    every slot from the explicit inputs before any read, so internal
    retraces across requests are safe. Compiled callables therefore
    survive cache-list turnover between requests: the steady state is
    zero traces per request.
    """

    def __init__(self, model: Any) -> None:
        self.model_ref = weakref.ref(model)
        self.disabled = False
        self._failures = 0
        self._compiled: dict[tuple[int, int], Any] = {}
        self._spec: list[tuple[int, str]] | None = None
        self._shadow: list[Any] | None = None
        self._shadow_sig: tuple | None = None
        self.stats: dict[str, Any] = {
            "calls": 0,
            "compiled_calls": 0,
            "traces": 0,
            "fallback_reasons": {},
        }

    # -- dispatch -----------------------------------------------------------

    def forward_verify(self, input_ids, cache, n_confirmed: int):
        """Compiled verify forward, or ``None`` when the caller must run
        the eager path for this round.

        On success the real cache entries are already mirror-committed
        (keys/values/offset advanced by S; GDN states + rollback_state
        snapshots installed), exactly as the eager forward would have
        left them.
        """
        self.stats["calls"] += 1
        if self.disabled:
            return self._miss("disabled")
        shape = getattr(input_ids, "shape", None)
        if shape is None or len(shape) != 2 or int(shape[0]) != 1:
            return self._miss("input_shape")
        S = int(shape[1])
        if S < 2 or not (0 < n_confirmed < S):
            return self._miss("window_shape")
        reason = self._eligibility(cache, S)
        if reason is not None:
            return self._miss(reason)
        try:
            self._ensure_capacity(cache, S)
            self._ensure_shadow(cache)
            state_in = self._read_state(cache)
            key = (S, int(n_confirmed))
            fn = self._compiled.get(key)
            if fn is None:
                fn = mx.compile(self._make_verify_step(S, int(n_confirmed)))
                self._compiled[key] = fn
            # Exactness boundary ("pre", MTPLX lesson): materialize any
            # pending producer graphs of the state leaves with the eager
            # kernels before they cross into compiled execution.
            mx.async_eval(*state_in)
            outputs = fn(input_ids, *state_in)
            logits, hidden, state_out = self._unpack(outputs, S, int(n_confirmed))
            # Buffer-safety boundary ("post", MTPLX lesson): schedule the
            # compiled graph while the real cache still references every
            # input leaf, so no input buffer can be donated/reused while
            # the graph is pending.
            mx.async_eval(*outputs)
        except Exception as exc:  # noqa: BLE001 — any failure demotes to eager
            self._failures += 1
            if self._failures >= 3:
                self.disabled = True
            logger.warning(
                "[MTP-compiled-verify] eager fallback (%s: %s)%s",
                type(exc).__name__,
                exc,
                " — bank disabled after 3 consecutive failures"
                if self.disabled
                else "",
            )
            return self._miss(f"exception:{type(exc).__name__}")
        self._failures = 0
        self.stats["compiled_calls"] += 1
        self._commit(cache, state_out, S, int(n_confirmed))
        self._clear_shadow_refs()
        return logits, hidden

    def _miss(self, reason: str):
        reasons = self.stats["fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return None

    # -- preconditions ------------------------------------------------------

    def _eligibility(self, cache, S: int) -> str | None:
        if cache is None:
            return "no_cache"
        from mlx_lm.models.cache import ArraysCache, KVCache

        spec: list[tuple[int, str]] = []
        for idx, entry in enumerate(cache):
            # Exact-type checks: subclasses (BatchKVCache, QuantizedKVCache,
            # rotating variants) have different update/mask semantics and
            # must stay on the eager path.
            if type(entry) is KVCache:
                if entry.keys is None or entry.values is None:
                    return "kv_empty"
                if not isinstance(entry.offset, int):
                    return "kv_offset_not_int"
                spec.append((idx, _KIND_FA))
            elif type(entry) is ArraysCache:
                if len(entry.cache) != 2:
                    return "gdn_slots"
                if entry.cache[0] is None or entry.cache[1] is None:
                    return "gdn_empty"
                if entry.lengths is not None or entry.left_padding is not None:
                    return "gdn_batched"
                spec.append((idx, _KIND_GDN))
            else:
                return f"unsupported:{type(entry).__name__}"
        if not any(kind == _KIND_FA for _idx, kind in spec):
            return "no_full_attn"
        self._spec = spec
        return None

    def _ensure_capacity(self, cache, S: int) -> None:
        """Pre-grow real KV buffers so the compiled graph keeps one shape.

        Growth mirrors the stock ``update_and_fetch`` concat (zero-filled
        tail, step-aligned) with an extra reserve so retraces are rare.
        Content within ``[:offset]`` is untouched, and eager consumers
        slice to ``offset``, so padding is invisible to numerics.
        """
        reserve = _growth_reserve()
        for idx, kind in self._spec or []:
            if kind != _KIND_FA:
                continue
            entry = cache[idx]
            step = int(getattr(entry, "step", 256) or 256)
            capacity = int(entry.keys.shape[2])
            needed = int(entry.offset) + S
            if needed <= capacity:
                continue
            target = needed + reserve
            new_capacity = ((target + step - 1) // step) * step
            grow = new_capacity - capacity
            B, n_kv, _, k_dim = entry.keys.shape
            v_dim = entry.values.shape[3]
            entry.keys = mx.concatenate(
                [entry.keys, mx.zeros((B, n_kv, grow, k_dim), entry.keys.dtype)],
                axis=2,
            )
            entry.values = mx.concatenate(
                [entry.values, mx.zeros((B, n_kv, grow, v_dim), entry.values.dtype)],
                axis=2,
            )

    # -- shadow cache ---------------------------------------------------------

    def _ensure_shadow(self, cache) -> None:
        sig = tuple(id(entry) for entry in cache)
        if self._shadow is not None and sig == self._shadow_sig:
            return
        from mlx_lm.models.cache import ArraysCache

        shadow: list[Any] = [None] * len(cache)
        for idx, kind in self._spec or []:
            shadow[idx] = _ShadowKVCache() if kind == _KIND_FA else ArraysCache(size=2)
        self._shadow = shadow
        self._shadow_sig = sig

    def _clear_shadow_refs(self) -> None:
        """Drop trace-time leaf references held by the shadow twins.

        The traced body re-seeds every slot from the explicit inputs
        before any read, so whatever the twins hold between calls is
        dead weight (post-trace tracers, or nothing on replays).
        """
        for entry in self._shadow or []:
            if entry is None:
                continue
            if isinstance(entry, _ShadowKVCache):
                entry.keys = None
                entry.values = None
                entry.offset = None
            else:
                entry.cache[0] = None
                entry.cache[1] = None
                entry.rollback_state = None

    # -- compiled function ------------------------------------------------------

    def _make_verify_step(self, S: int, n_confirmed: int):
        spec = list(self._spec or [])
        boundaries = list(range(n_confirmed, S))
        bank = self

        def verify_step(input_ids, *state_in):
            # Python body runs at trace time only; replays skip it. It
            # reads the bank's LIVE shadow list, so internal retraces
            # (leaf-shape changes) always land on current containers.
            bank.stats["traces"] += 1
            shadow = bank._shadow
            model = bank.model_ref()
            if model is None:
                raise RuntimeError("model was garbage collected")
            # (1) Re-seed firewall: every shadow leaf is assigned from
            # the explicit inputs BEFORE any read.
            pos = 0
            for idx, kind in spec:
                entry = shadow[idx]
                if kind == _KIND_FA:
                    entry.seed(state_in[pos], state_in[pos + 1], state_in[pos + 2])
                    pos += 3
                else:
                    entry.cache[0] = state_in[pos]
                    entry.cache[1] = state_in[pos + 1]
                    entry.rollback_state = None
                    pos += 2
            # (2) The existing forward — including the GatedDeltaNet
            # chunk-split — against shadow containers only.
            logits, hidden = model(
                input_ids,
                cache=shadow,
                return_hidden=True,
                n_confirmed=n_confirmed,
            )
            # (3) Read every leaf back out and return it explicitly.
            state_out: list[Any] = []
            for idx, kind in spec:
                entry = shadow[idx]
                if kind == _KIND_FA:
                    state_out.extend((entry.keys, entry.values))
                else:
                    snaps = entry.rollback_state
                    if not isinstance(snaps, dict) or sorted(snaps) != boundaries:
                        raise RuntimeError(
                            "GDN chunk-split produced snapshots "
                            f"{sorted(snaps) if isinstance(snaps, dict) else snaps!r}, "
                            f"expected boundaries {boundaries}"
                        )
                    state_out.extend((entry.cache[0], entry.cache[1]))
                    for b in boundaries:
                        state_out.extend((snaps[b][0], snaps[b][1]))
            return (logits, hidden, *state_out)

        return verify_step

    # -- state movement -----------------------------------------------------------

    def _read_state(self, cache) -> list[Any]:
        leaves: list[Any] = []
        for idx, kind in self._spec or []:
            entry = cache[idx]
            if kind == _KIND_FA:
                leaves.extend(
                    (entry.keys, entry.values, mx.array(entry.offset, dtype=mx.int32))
                )
            else:
                leaves.extend((entry.cache[0], entry.cache[1]))
        return leaves

    def _unpack(self, outputs, S: int, n_confirmed: int):
        n_bound = S - n_confirmed
        n_state = sum(
            2 if kind == _KIND_FA else 2 + 2 * n_bound for _idx, kind in self._spec or []
        )
        expected = 2 + n_state
        if len(outputs) != expected:
            raise ValueError(
                f"compiled verify returned {len(outputs)} leaves, expected {expected}"
            )
        return outputs[0], outputs[1], list(outputs[2:])

    def _commit(self, cache, state_out: list[Any], S: int, n_confirmed: int) -> None:
        """Mirror the compiled outputs into the real stock cache entries.

        Leaves the entries exactly as the eager forward would have:
        KV buffers hold the verify rows at ``[offset:offset+S)`` with the
        Python-int offset advanced by S (so the reject path's
        ``KVCache.trim`` keeps working unchanged), and each ArraysCache
        carries the fresh ``rollback_state`` snapshot dict the
        generator's ``_rollback_draft``/``_clear_rollback`` expect.
        """
        boundaries = list(range(n_confirmed, S))
        pos = 0
        for idx, kind in self._spec or []:
            entry = cache[idx]
            if kind == _KIND_FA:
                entry.keys = state_out[pos]
                entry.values = state_out[pos + 1]
                entry.offset = int(entry.offset) + S
                pos += 2
            else:
                entry.cache[0] = state_out[pos]
                entry.cache[1] = state_out[pos + 1]
                pos += 2
                snaps: dict[int, tuple] = {}
                for b in boundaries:
                    snaps[b] = (state_out[pos], state_out[pos + 1])
                    pos += 2
                entry.rollback_state = snaps


# One bank per live model object. Keyed by id() with an identity re-check
# (id() can be recycled after a model swap frees the old one).
_BANKS: dict[int, CompiledVerify] = {}


def get_bank(model: Any) -> CompiledVerify:
    key = id(model)
    bank = _BANKS.get(key)
    if bank is None or bank.model_ref() is not model:
        bank = CompiledVerify(model)
        _BANKS[key] = bank
    return bank
