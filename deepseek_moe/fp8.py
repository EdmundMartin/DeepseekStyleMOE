"""Simulated FP8 (E4M3) quantization with DeepSeek-V3's fine-grained scaling.

V3 quantizes activations per 1x128 tile (per token, per 128 channels) and
weights per 128x128 block, each with its own scale. That keeps a single
outlier from wrecking the precision of an entire tensor.

Training uses *fake* quantization: values are rounded through float8 and cast
back, and a straight-through estimator lets gradients flow as if no rounding
happened. ``FP8Linear.freeze_fp8()`` converts the weight to real float8
storage for inference, cutting its memory to a quarter of fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


def _pad_to(x: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    pad = (-x.shape[dim]) % multiple
    if pad == 0:
        return x
    pad_spec = [0, 0] * (x.dim() - 1 - dim % x.dim()) + [0, pad]
    return F.pad(x, pad_spec)


def quantize_blockwise(w: torch.Tensor, block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight with one scale per (block x block) tile.

    Returns (fp8 weight with the original shape, fp32 scales of shape [ceil(out/b), ceil(in/b)]).
    """
    out_f, in_f = w.shape
    wp = _pad_to(_pad_to(w.float(), 0, block), 1, block)
    ob, ib = wp.shape[0] // block, wp.shape[1] // block
    tiles = wp.view(ob, block, ib, block)
    amax = tiles.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    q = (tiles / scale).to(FP8_DTYPE)
    q = q.view(ob * block, ib * block)[:out_f, :in_f]
    return q, scale.view(ob, ib)


def dequantize_blockwise(q: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    out_f, in_f = q.shape
    s = scale.repeat_interleave(block, 0).repeat_interleave(block, 1)[:out_f, :in_f]
    return q.float() * s


def fake_quant_weight(w: torch.Tensor, block: int) -> torch.Tensor:
    q, scale = quantize_blockwise(w.detach(), block)
    wq = dequantize_blockwise(q, scale, block).to(w.dtype)
    return w + (wq - w).detach()  # straight-through estimator


def fake_quant_activation(x: torch.Tensor, block: int) -> torch.Tensor:
    """Per-token, per-``block``-channel tile quantization of the last dim."""
    d = x.shape[-1]
    xp = _pad_to(x.detach().float(), -1, block)
    tiles = xp.view(*xp.shape[:-1], xp.shape[-1] // block, block)
    amax = tiles.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    xq = ((tiles / scale).to(FP8_DTYPE).float() * scale).view(xp.shape)[..., :d].to(x.dtype)
    return x + (xq - x).detach()


class FP8Linear(nn.Linear):
    """nn.Linear whose GEMM inputs are rounded to FP8 with fine-grained scales."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, block_size: int = 128):
        super().__init__(in_features, out_features, bias=bias)
        self.block_size = block_size
        self.register_buffer("weight_scale", None)

    @property
    def frozen(self) -> bool:
        return self.weight.dtype == FP8_DTYPE

    def compute_weight(self) -> torch.Tensor:
        """The weight actually used in the matmul (quantize -> dequantize)."""
        if self.frozen:
            return dequantize_blockwise(self.weight, self.weight_scale, self.block_size)
        return fake_quant_weight(self.weight, self.block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = fake_quant_activation(x, self.block_size)
        w = self.compute_weight().to(x.dtype)
        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def freeze_fp8(self) -> None:
        """Store the weight as real float8 + block scales (inference only)."""
        if self.frozen:
            return
        q, scale = quantize_blockwise(self.weight, self.block_size)
        self.weight = nn.Parameter(q, requires_grad=False)
        self.weight_scale = scale


def make_linear(in_features: int, out_features: int, use_fp8: bool, block_size: int) -> nn.Linear:
    if use_fp8:
        return FP8Linear(in_features, out_features, block_size=block_size)
    return nn.Linear(in_features, out_features, bias=False)


def linear_weight(layer: nn.Linear) -> torch.Tensor:
    """Effective weight of a (possibly FP8) linear layer."""
    return layer.compute_weight() if isinstance(layer, FP8Linear) else layer.weight
