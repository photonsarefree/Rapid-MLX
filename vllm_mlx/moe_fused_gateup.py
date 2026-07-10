"""Fuse the MoE gate + up expert projections into one gather_qmm.

Engine-integrated version of the previously external pipeline patch.
Enabled by default; RAPID_MLX_MOE_FUSED_GATEUP=0 disables. Token-exact
(pure dispatch reordering, fingerprint-verified); mutually idempotent with
the external patch via the _optim_fused_gateup class sentinel.

Stock SwitchGLU (mlx-lm switch_layers.py) does TWO gather_qmm dispatches over
the same x and the same expert indices — one for up_proj, one for gate_proj —
then SwiGLU-combines them. Concatenating the two experts' weights along the
output dim lets us do ONE gather that emits 2*hidden, then split. Half the
expert-gather dispatches on the 35B's 256-expert layers.

The fusion is a pure reordering, so greedy output must be byte-identical — the
compounding bench asserts this via the fingerprint. The fused weight is built
lazily on first forward and cached on the module; the originals are dropped to
reclaim memory (net zero, no extra resident copy).

Mechanism note: QuantizedSwitchLinear packs along the INPUT dim (last axis), so
concatenating gate/up along the OUTPUT dim (axis=1) of weight/scales/biases is
a clean, correct concat that needs no re-quantization.
"""

from __future__ import annotations


def apply() -> bool:
    import mlx.core as mx
    from mlx_lm.models import switch_layers as sl

    if getattr(sl.SwitchGLU, "_optim_fused_gateup", False):
        return False

    def _fuse(gp, up):
        """Concatenate two (Quantized)SwitchLinear along output_dims -> one module."""
        if hasattr(gp, "scales"):  # QuantizedSwitchLinear
            fused = sl.QuantizedSwitchLinear.__new__(sl.QuantizedSwitchLinear)
            import mlx.nn as nn
            nn.Module.__init__(fused)
            fused.weight = mx.concatenate([gp["weight"], up["weight"]], axis=1)
            fused.scales = mx.concatenate([gp["scales"], up["scales"]], axis=1)
            gb, ub = gp.get("biases"), up.get("biases")
            fused.biases = mx.concatenate([gb, ub], axis=1) if gb is not None else None
            if "bias" in gp:
                fused.bias = mx.concatenate([gp["bias"], up["bias"]], axis=1)
            fused.group_size, fused.bits, fused.mode = gp.group_size, gp.bits, gp.mode
            fused.freeze()
        else:  # plain SwitchLinear
            fused = sl.SwitchLinear.__new__(sl.SwitchLinear)
            import mlx.nn as nn
            nn.Module.__init__(fused)
            fused.weight = mx.concatenate([gp["weight"], up["weight"]], axis=1)
            if "bias" in gp:
                fused.bias = mx.concatenate([gp["bias"], up["bias"]], axis=1)
            fused.freeze()
        return fused

    def fused_call(self, x, indices):
        if not hasattr(self, "_gate_up"):
            self._gate_up = _fuse(self.gate_proj, self.up_proj)
            # drop originals to reclaim memory (fused replaces them)
            self.gate_proj = None
            self.up_proj = None

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = sl._gather_sort(x, indices)
        gu = self._gate_up(x, idx, sorted_indices=do_sort)  # [..., 2*hidden]
        h = gu.shape[-1] // 2
        x_gate, x_up = gu[..., :h], gu[..., h:]
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = sl._scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)

    sl.SwitchGLU.__call__ = fused_call
    sl.SwitchGLU._optim_fused_gateup = True
    return True
