"""Multi-head Latent Attention (MLA), as in DeepSeek-V2/V3.

Standard MHA caches full K and V for every head: 2 * n_heads * head_dim values
per token per layer. MLA instead caches

  * c_KV  - a low-rank latent (kv_lora_rank values) from which every head's
            keys and values are reconstructed by an up-projection, and
  * k_pe  - one small RoPE key (qk_rope_head_dim values) shared by all heads.

RoPE is "decoupled": position information lives only in the extra q_pe/k_pe
dims. The content dims carry no rotation, which is what lets the key
up-projection W_UK be absorbed into the query at inference time, so we never
materialize per-head keys or values from the cache at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .fp8 import linear_weight, make_linear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent pairs of the last dim. x: [B, T, H, D], cos/sin: [T, D/2]."""
    x1, x2 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)


class MLA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.nope_dim = cfg.qk_nope_head_dim
        self.rope_dim = cfg.qk_rope_head_dim
        self.v_dim = cfg.v_head_dim
        self.kv_lora_rank = cfg.kv_lora_rank
        self.scale = 1.0 / math.sqrt(cfg.qk_head_dim)
        lin = lambda i, o: make_linear(i, o, cfg.use_fp8, cfg.fp8_block_size)

        # Queries: optionally compressed too (reduces activation memory in training).
        self.q_lora_rank = cfg.q_lora_rank
        if cfg.q_lora_rank > 0:
            self.wq_a = lin(cfg.dim, cfg.q_lora_rank)
            self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
            self.wq_b = lin(cfg.q_lora_rank, cfg.n_heads * cfg.qk_head_dim)
        else:
            self.wq = lin(cfg.dim, cfg.n_heads * cfg.qk_head_dim)

        # Joint KV down-projection -> [c_KV | k_pe]
        self.wkv_a = lin(cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim)
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, cfg.norm_eps)
        # Up-projection c_KV -> per-head [k_nope | v]  (holds W_UK and W_UV)
        self.wkv_b = lin(cfg.kv_lora_rank, cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim))
        self.wo = lin(cfg.n_heads * cfg.v_head_dim, cfg.dim)

    def _queries(self, x: torch.Tensor, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))) if self.q_lora_rank > 0 else self.wq(x)
        q = q.view(B, T, self.n_heads, self.nope_dim + self.rope_dim)
        q_nope, q_pe = q.split([self.nope_dim, self.rope_dim], dim=-1)
        return q_nope, apply_rope(q_pe, cos, sin)

    def _latents(self, x: torch.Tensor, cos, sin) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.wkv_a(x)
        c_kv, k_pe = kv.split([self.kv_lora_rank, self.rope_dim], dim=-1)
        k_pe = apply_rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)  # [B, T, rope_dim]
        return self.kv_norm(c_kv), k_pe

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """
        x: [B, T, dim]; cos/sin already sliced to this chunk's absolute positions.
        cache: None for training (naive path), or a dict that is filled/extended in
               place with the compressed latents for incremental decoding.
        """
        q_nope, q_pe = self._queries(x, cos, sin)
        c_kv, k_pe = self._latents(x, cos, sin)
        if cache is None:
            return self._forward_naive(q_nope, q_pe, c_kv, k_pe)

        if "c_kv" in cache:
            c_kv = torch.cat([cache["c_kv"], c_kv], dim=1)
            k_pe = torch.cat([cache["k_pe"], k_pe], dim=1)
        cache["c_kv"], cache["k_pe"] = c_kv, k_pe
        return self._forward_absorbed(q_nope, q_pe, c_kv, k_pe)

    def _forward_naive(self, q_nope, q_pe, c_kv, k_pe) -> torch.Tensor:
        """Up-project the latent to full per-head K/V, then ordinary attention."""
        B, T, H, _ = q_nope.shape
        kv = self.wkv_b(c_kv).view(B, T, H, self.nope_dim + self.v_dim)
        k_nope, v = kv.split([self.nope_dim, self.v_dim], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.unsqueeze(2).expand(B, T, H, self.rope_dim)], dim=-1)
        # The fused (flash) SDPA kernel needs V's head dim to match Q/K's; otherwise PyTorch
        # falls back to the math path, which materializes the full T x T attention matrix.
        # Zero-padding V and slicing the output is exact and ~3.5x faster on CPU.
        pad = q.shape[-1] - self.v_dim
        if pad > 0:
            v = F.pad(v, (0, pad))
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, scale=self.scale
        )[..., : self.v_dim]
        return self.wo(out.transpose(1, 2).reshape(B, T, H * self.v_dim))

    def _forward_absorbed(self, q_nope, q_pe, c_kv, k_pe) -> torch.Tensor:
        """Attend directly in latent space using the weight-absorption trick.

        score = q_nope . (W_UK c) + q_pe . k_pe = (W_UK^T q_nope) . c + q_pe . k_pe
        out   = W_UV (sum_t p_t c_t)
        """
        B, S, H, _ = q_nope.shape
        T = c_kv.shape[1]
        w = linear_weight(self.wkv_b).view(H, self.nope_dim + self.v_dim, self.kv_lora_rank)
        w_uk, w_uv = w[:, : self.nope_dim], w[:, self.nope_dim :]

        q_lat = torch.einsum("bshd,hdr->bshr", q_nope, w_uk)
        scores = torch.einsum("bshr,btr->bsht", q_lat, c_kv) + torch.einsum("bshd,btd->bsht", q_pe, k_pe)
        scores = scores.float() * self.scale
        # New queries sit at absolute positions T-S .. T-1.
        mask = torch.ones(S, T, dtype=torch.bool, device=scores.device).tril(diagonal=T - S)
        scores = scores.masked_fill(~mask[None, :, None, :], float("-inf"))
        p = scores.softmax(dim=-1).to(c_kv.dtype)

        o_lat = torch.einsum("bsht,btr->bshr", p, c_kv)
        out = torch.einsum("bshr,hvr->bshv", o_lat, w_uv)
        return self.wo(out.reshape(B, S, H * self.v_dim))
