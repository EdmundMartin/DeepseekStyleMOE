"""DeepSeekMoE with auxiliary-loss-free load balancing (DeepSeek-V3).

Key ideas:
  * Fine-grained experts: many narrow experts, several activated per token,
    giving far more combinations of specialists than a few wide experts.
  * Shared experts: always-on experts that absorb common knowledge so routed
    experts can specialize.
  * Sigmoid affinities, normalized over the selected top-k.
  * Group-limited routing: experts are split into groups (nodes in V3) and each
    token may only pick from its best ``n_limited_groups`` groups, bounding
    cross-device communication.
  * Aux-loss-free balancing: a per-expert bias is added to affinities *only for
    top-k selection*. After each step, overloaded experts have their bias
    lowered and underloaded ones raised. Gating weights still use the unbiased
    scores, so balancing doesn't distort the model's output.
  * A tiny complementary sequence-wise balance loss guards against extreme
    imbalance within a single sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .fp8 import make_linear
from .runtime import full_precision


class _GatherRows(torch.autograd.Function):
    """x.index_select(0, idx) with a backward that is one zero-fill plus one index_add_.

    Plain x[idx] backpropagates through index_put_ with accumulation, which is slow on CPU;
    this keeps the expert-input gradient to a single scatter for all experts together.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.n_rows = x.shape[0]
        return x.index_select(0, idx)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (idx,) = ctx.saved_tensors
        return grad.new_zeros(ctx.n_rows, grad.shape[1]).index_add_(0, idx, grad), None


# On CPU the per-expert loop is fastest for a few experts; with many small experts its
# per-expert full-size gradient buffers dominate and the sorted single-gather path wins
# (measured: 12 experts 0.83x, 16 experts 1.02x, 80 experts 1.14x for sorted vs loop).
SORTED_DISPATCH_MIN_EXPERTS_CPU = 32


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, use_fp8: bool = False, block_size: int = 128):
        super().__init__()
        self.w1 = make_linear(dim, hidden, use_fp8, block_size)  # gate
        self.w3 = make_linear(dim, hidden, use_fp8, block_size)  # up
        self.w2 = make_linear(hidden, dim, use_fp8, block_size)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_routed_experts
        self.top_k = cfg.n_activated_experts
        self.n_groups = cfg.n_expert_groups
        self.topk_groups = cfg.n_limited_groups
        self.route_scale = cfg.route_scale
        # The gate stays in high precision, as in V3.
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        # Balancing bias: not trained by gradient, updated by update_bias().
        self.register_buffer("bias", torch.zeros(cfg.n_routed_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [N, dim] -> (weights [N, k], indices [N, k], scores [N, E])."""
        with full_precision(x):  # router stays fp32 even under bf16 autocast
            scores = torch.sigmoid(F.linear(x.float(), self.weight.float()))
        biased = scores + self.bias

        if self.n_groups > 1:
            # Rank groups by the sum of their top-2 biased scores; keep the best few.
            grouped = biased.view(-1, self.n_groups, self.n_experts // self.n_groups)
            k_in_group = min(2, grouped.shape[-1])
            group_scores = grouped.topk(k_in_group, dim=-1).values.sum(-1)
            top_groups = group_scores.topk(self.topk_groups, dim=-1).indices
            keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, top_groups, True)
            biased = grouped.masked_fill(~keep.unsqueeze(-1), float("-inf")).flatten(1)

        indices = biased.topk(self.top_k, dim=-1).indices
        weights = scores.gather(1, indices)  # unbiased affinities
        weights = weights / weights.sum(-1, keepdim=True) * self.route_scale
        return weights.type_as(x), indices, scores


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.gate = Gate(cfg)
        self.experts = nn.ModuleList(
            SwiGLU(cfg.dim, cfg.moe_hidden_dim, cfg.use_fp8, cfg.fp8_block_size)
            for _ in range(cfg.n_routed_experts)
        )
        self.shared = (
            SwiGLU(cfg.dim, cfg.n_shared_experts * cfg.moe_hidden_dim, cfg.use_fp8, cfg.fp8_block_size)
            if cfg.n_shared_experts > 0
            else None
        )
        # Tokens routed to each expert since the last bias update.
        self.register_buffer("expert_load", torch.zeros(cfg.n_routed_experts), persistent=False)
        self.aux_loss: torch.Tensor | None = None
        # Optional [B, T] bool mask of real (non-padding) tokens, set by the model for one forward.
        # Padding still flows through the experts but is excluded from load stats and the aux loss.
        self.token_mask: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        weights, indices, scores = self.gate(flat)

        counts = torch.bincount(indices.flatten(), minlength=self.cfg.n_routed_experts)
        mask = self.token_mask[:, :T] if self.token_mask is not None else None  # MTP blocks see shorter T
        if self.training:
            real = indices if mask is None else indices[mask.reshape(-1)]
            load = counts if mask is None else torch.bincount(real.flatten(), minlength=self.cfg.n_routed_experts)
            self.expert_load += load.to(self.expert_load.dtype)
            self.aux_loss = self._sequence_aux_loss(indices.view(B, T, -1), scores.view(B, T, -1), mask)
        else:
            self.aux_loss = None

        if flat.device.type == "cpu" and self.cfg.n_routed_experts < SORTED_DISPATCH_MIN_EXPERTS_CPU:
            out = self._dispatch_loop(flat, weights, indices, counts)
        else:
            out = self._dispatch_sorted(flat, weights, indices, counts)

        if self.shared is not None:
            out = out + self.shared(flat)
        return out.view(B, T, D)

    def _dispatch_loop(self, flat, weights, indices, counts) -> torch.Tensor:
        """Per-expert gather/scatter. Fastest on CPU in benchmarks."""
        out = torch.zeros_like(flat)
        for e, n in enumerate(counts.tolist()):
            if n == 0:
                continue
            tok, slot = torch.where(indices == e)
            out.index_add_(0, tok, self.experts[e](flat[tok]) * weights[tok, slot, None])
        return out

    def _dispatch_sorted(self, flat, weights, indices, counts) -> torch.Tensor:
        """Sort token-expert pairs by expert once, run each expert on a contiguous slice.

        Replaces E scans of the routing table with one argsort, one gather and one scatter, and
        needs a single host sync per layer (for the split sizes). Used on GPUs and for many-expert
        models on CPU.
        """
        K = indices.shape[1]
        flat_idx = indices.flatten()
        order = flat_idx.argsort(stable=True)
        tok = order // K
        xs = _GatherRows.apply(flat, tok)
        outs = [self.experts[e](chunk) for e, chunk in enumerate(xs.split(counts.tolist())) if chunk.shape[0]]
        y = torch.cat(outs) * weights.flatten()[order, None]
        return torch.zeros(flat.shape, dtype=y.dtype, device=flat.device).index_add_(0, tok, y).to(flat.dtype)

    def _sequence_aux_loss(self, indices: torch.Tensor, scores: torch.Tensor,
                           mask: torch.Tensor | None = None) -> torch.Tensor:
        """L_bal = alpha * sum_i f_i * P_i, computed per sequence over real tokens and averaged."""
        B, T, K = indices.shape
        E = self.cfg.n_routed_experts
        w = torch.ones(B, T, device=indices.device) if mask is None else mask.to(scores.dtype)
        n = w.sum(1).clamp(min=1)  # real tokens per sequence
        one_hot = torch.zeros(B, T, E, device=indices.device).scatter_(2, indices, 1.0) * w[..., None]
        f = one_hot.sum(1) * (E / K) / n[:, None]  # [B, E]
        p = ((scores / scores.sum(-1, keepdim=True)) * w[..., None]).sum(1) / n[:, None]  # [B, E]
        return self.cfg.seq_aux_loss_alpha * (f * p).sum(-1).mean()

    @torch.no_grad()
    def update_bias(self, update: bool = True) -> torch.Tensor:
        """Nudge each expert's bias toward balanced load (unless update=False); returns the observed load."""
        load = self.expert_load.clone()
        if update and load.sum() > 0:
            self.gate.bias += self.cfg.bias_update_speed * torch.sign(load.mean() - load)
        self.expert_load.zero_()
        return load
