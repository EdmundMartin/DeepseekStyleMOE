"""The full DeepSeek-style model: MLA + DeepSeekMoE blocks with MTP heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .fp8 import FP8Linear
from .mla import MLA, RMSNorm, precompute_rope
from .moe import Gate, MoE, SwiGLU


def sample_next(
    logits: torch.Tensor,
    context: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Pick the next token from last-position logits [B, V]; returns [B, 1].

    Order: repetition penalty -> temperature -> top-k -> top-p (nucleus) -> sample.
    temperature <= 0 means greedy (after the repetition penalty).
    """
    logits = logits.float()
    if repetition_penalty != 1.0:
        # CTRL-style penalty on every token already in the context: shrink positive
        # logits and push negative ones further down, so repeats get less likely.
        prev = logits.gather(1, context)
        prev = torch.where(prev > 0, prev / repetition_penalty, prev * repetition_penalty)
        logits = logits.scatter(1, context, prev)
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        v, _ = logits.topk(min(top_k, logits.shape[-1]))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        # Drop tokens once the probability mass *before* them already exceeds top_p
        # (so the single most likely token is always kept).
        drop = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, order, sorted_logits)
    return torch.multinomial(logits.softmax(-1), 1)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy ignoring targets of -100 (SFT prompt tokens / padding).

    Returns a zero that still carries gradient when every target is masked, where
    F.cross_entropy would return NaN.
    """
    if (targets != -100).any():
        return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
    return logits.sum() * 0.0


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_moe: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = MLA(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = MoE(cfg) if use_moe else SwiGLU(cfg.dim, cfg.dense_hidden_dim, cfg.use_fp8, cfg.fp8_block_size)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.ffn(self.ffn_norm(x))


class MTPModule(nn.Module):
    """One sequential Multi-Token Prediction depth (DeepSeek-V3 section 2.2).

    Combines the previous depth's hidden state for position i with the
    embedding of token i+k, runs one transformer block, and predicts token
    i+k+1 through the *shared* output head. Keeping predictions sequential
    (rather than parallel independent heads) preserves the causal chain.
    """

    def __init__(self, cfg: ModelConfig, use_moe: bool):
        super().__init__()
        self.hidden_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.emb_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        self.block = Block(cfg, use_moe)
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)

    def forward(self, h_prev, emb_next, cos, sin):
        h = self.proj(torch.cat([self.hidden_norm(h_prev), self.emb_norm(emb_next)], dim=-1))
        return self.block(h, cos, sin)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class DeepSeekMoEModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg, use_moe=i >= cfg.n_dense_layers) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)  # kept in high precision
        has_moe = cfg.n_layers > cfg.n_dense_layers
        self.mtp = nn.ModuleList(MTPModule(cfg, use_moe=has_moe) for _ in range(cfg.n_mtp_modules))

        cos, sin = precompute_rope(cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(m, (nn.Linear, nn.Embedding, Gate)):
            nn.init.normal_(m.weight, std=std)

    # ------------------------------------------------------------------ forward

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                token_mask: torch.Tensor | None = None) -> ModelOutput:
        """Training / evaluation forward. idx, targets: [B, T] (targets = idx shifted by 1).

        token_mask: optional [B, T] bool of real (non-padding) positions; padding is excluded
        from MoE load statistics and the balance loss (SFT batches are right-padded).
        """
        for m in self.moe_layers():
            m.token_mask = token_mask
        try:
            return self._forward(idx, targets)
        finally:
            for m in self.moe_layers():
                m.token_mask = None

    def _forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> ModelOutput:
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"sequence length {T} > max_seq_len {self.cfg.max_seq_len}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]

        emb = self.embed(idx)
        h = emb
        for layer in self.layers:
            h = layer(h, cos, sin)
        logits = self.head(self.norm(h))
        if targets is None:
            return ModelOutput(logits)

        main_loss = masked_cross_entropy(logits, targets)
        metrics = {"lm_loss": main_loss.item()}
        loss = main_loss

        # Multi-token prediction: depth k predicts token i+k+1 from position i.
        # MTP is a training signal only; evaluation reports the main next-token loss.
        mtp_losses = []
        h_prev = h
        for k, mod in enumerate(self.mtp if self.training else [], start=1):
            if T - k <= 0:
                break
            h_prev = mod(h_prev[:, : T - k], emb[:, k:], cos[: T - k], sin[: T - k])
            mtp_logits = self.head(mod.norm(h_prev))
            mtp_losses.append(masked_cross_entropy(mtp_logits, targets[:, k:]))
        if mtp_losses:
            mtp_loss = torch.stack(mtp_losses).mean()
            loss = loss + self.cfg.mtp_loss_weight * mtp_loss
            metrics["mtp_loss"] = mtp_loss.item()

        aux = [m.aux_loss for m in self.moe_layers() if m.aux_loss is not None]
        if aux:
            aux_loss = torch.stack(aux).sum()
            loss = loss + aux_loss
            metrics["aux_loss"] = aux_loss.item()
        return ModelOutput(logits, loss, metrics)

    def forward_cached(self, idx: torch.Tensor, caches: list[dict], start_pos: int) -> torch.Tensor:
        """Incremental forward for generation; returns logits for the last position."""
        T = idx.shape[1]
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        h = self.embed(idx)
        for layer, cache in zip(self.layers, caches):
            h = layer(h, cos, sin, cache)
        return self.head(self.norm(h[:, -1]))

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        stop_ids: set[int] | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """Sample using the MLA latent KV cache (the cache holds only c_KV and k_pe)."""
        self.eval()
        window = self.cfg.max_seq_len
        caches: list[dict] = []
        pos = 0
        for _ in range(max_new_tokens):
            if not caches or pos >= window:
                # (Re)prefill. When the window is full, restart from the most recent half.
                ctx = idx[:, -(window // 2 if pos >= window else window) :]
                caches = [{} for _ in self.layers]
                logits = self.forward_cached(ctx, caches, 0)
                pos = ctx.shape[1]
            else:
                logits = self.forward_cached(idx[:, -1:], caches, pos)
                pos += 1
            nxt = sample_next(logits, idx, temperature, top_k, top_p, repetition_penalty)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_ids and idx.shape[0] == 1 and nxt.item() in stop_ids:
                break
        return idx

    # ------------------------------------------------------------------ helpers

    @torch.no_grad()
    def resize_vocab(self, new_size: int) -> None:
        """Grow the embedding and output head (e.g. after adding chat special tokens).

        New rows start at the mean of the existing rows, a neutral starting point
        that keeps the new tokens from dominating the softmax before fine-tuning.
        """
        old = self.cfg.vocab_size
        if new_size <= old:
            return
        emb = nn.Embedding(new_size, self.cfg.dim).to(self.embed.weight)
        emb.weight[:old] = self.embed.weight
        emb.weight[old:] = self.embed.weight.mean(0)
        head = nn.Linear(self.cfg.dim, new_size, bias=False).to(self.head.weight)
        head.weight[:old] = self.head.weight
        head.weight[old:] = self.head.weight.mean(0)
        self.embed, self.head = emb, head
        self.cfg.vocab_size = new_size

    def moe_layers(self) -> list[MoE]:
        return [m for m in self.modules() if isinstance(m, MoE)]

    @torch.no_grad()
    def update_moe_biases(self, update: bool = True) -> dict[str, float]:
        """Aux-loss-free balancing step; call once after each optimizer step.

        update=False only reports (and resets) the load stats, leaving the biases frozen.
        """
        imbalance = []
        for m in self.moe_layers():
            load = m.update_bias(update)
            if load.sum() > 0:
                imbalance.append((load.max() / load.mean()).item())
        if not imbalance:
            return {}
        # Busiest expert / mean load, per MoE layer. The ceiling is n_routed / n_activated
        # (one expert chosen by every token), so compare against that.
        return {"max_load_ratio": max(imbalance), "mean_load_ratio": sum(imbalance) / len(imbalance)}

    def freeze_fp8(self) -> None:
        """Convert all FP8 linears to real float8 weight storage (inference only)."""
        for m in self.modules():
            if isinstance(m, FP8Linear):
                m.freeze_fp8()

    def param_counts(self) -> dict[str, int]:
        """Total params vs. params activated per token (main model, excluding MTP)."""
        main = [p for n, p in self.named_parameters() if not n.startswith("mtp.")]
        total = sum(p.numel() for p in main)
        per_expert = sum(p.numel() for p in self.layers[-1].ffn.experts[0].parameters()) if self.cfg.n_layers > self.cfg.n_dense_layers else 0
        n_moe = self.cfg.n_layers - self.cfg.n_dense_layers
        inactive = n_moe * (self.cfg.n_routed_experts - self.cfg.n_activated_experts) * per_expert
        mtp = sum(p.numel() for p in self.mtp.parameters())
        return {"total": total, "activated": total - inactive, "mtp": mtp}
