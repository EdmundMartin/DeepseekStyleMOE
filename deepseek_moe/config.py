"""Model configuration.

Every architectural knob from DeepSeek-V3 is exposed here so the model can be
scaled from a toy that trains on a laptop CPU up to something GPU-sized.
Use ``ModelConfig.from_preset("tiny", dim=384)`` to start from a preset and
override individual fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    # --- core transformer ---
    vocab_size: int = 256  # byte-level tokenizer by default
    max_seq_len: int = 1024
    dim: int = 256
    n_layers: int = 4
    n_dense_layers: int = 1  # first k layers use a dense FFN (V3 uses 3 of 61)
    dense_hidden_dim: int = 704
    norm_eps: float = 1e-6
    init_std: float = 0.02

    # --- Multi-head Latent Attention (MLA) ---
    n_heads: int = 4
    q_lora_rank: int = 0  # 0 disables query compression (V2-Lite style)
    kv_lora_rank: int = 64  # size of the cached latent c_KV
    qk_nope_head_dim: int = 32  # per-head dims that carry content
    qk_rope_head_dim: int = 16  # per-head dims that carry position (decoupled RoPE)
    v_head_dim: int = 32
    rope_theta: float = 10000.0

    # --- DeepSeekMoE ---
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2  # top-k routed experts per token
    moe_hidden_dim: int = 128  # fine-grained experts are narrow
    n_expert_groups: int = 1  # group-limited (node-limited) routing
    n_limited_groups: int = 1  # groups each token may route into
    route_scale: float = 1.0
    bias_update_speed: float = 1e-3  # gamma for auxiliary-loss-free balancing
    seq_aux_loss_alpha: float = 1e-4  # small complementary sequence-wise loss

    # --- Multi-Token Prediction (MTP) ---
    n_mtp_modules: int = 1  # D in the paper; 0 disables MTP
    mtp_loss_weight: float = 0.3  # lambda

    # --- FP8 (simulated) ---
    use_fp8: bool = False
    fp8_block_size: int = 128  # 1xB activation tiles, BxB weight blocks

    def __post_init__(self) -> None:
        assert 0 <= self.n_dense_layers <= self.n_layers
        assert self.qk_rope_head_dim % 2 == 0, "RoPE needs an even head dim"
        assert self.n_activated_experts <= self.n_routed_experts
        assert self.n_routed_experts % self.n_expert_groups == 0
        assert self.n_limited_groups <= self.n_expert_groups
        experts_per_group = self.n_routed_experts // self.n_expert_groups
        assert self.n_limited_groups * experts_per_group >= self.n_activated_experts, (
            "group-limited routing must leave at least top-k experts selectable"
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def param_estimate(self) -> dict[str, int]:
        """Exact parameter counts computed from the config, without building the model.

        Matches DeepSeekMoEModel.param_counts(); usable for presets far too big to instantiate.
        """
        d, H = self.dim, self.n_heads
        if self.q_lora_rank > 0:
            q = d * self.q_lora_rank + self.q_lora_rank + self.q_lora_rank * H * self.qk_head_dim
        else:
            q = d * H * self.qk_head_dim
        kv = d * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank
        kv += self.kv_lora_rank * H * (self.qk_nope_head_dim + self.v_head_dim)
        attn = q + kv + H * self.v_head_dim * d
        expert = 3 * d * self.moe_hidden_dim
        dense_block = 2 * d + attn + 3 * d * self.dense_hidden_dim
        moe_block = (2 * d + attn + self.n_routed_experts * d + self.n_routed_experts * expert
                     + self.n_shared_experts * expert)
        n_moe = self.n_layers - self.n_dense_layers
        total = 2 * self.vocab_size * d + d + self.n_dense_layers * dense_block + n_moe * moe_block
        inactive = n_moe * (self.n_routed_experts - self.n_activated_experts) * expert
        mtp_block = moe_block if n_moe > 0 else dense_block
        mtp = self.n_mtp_modules * (3 * d + 2 * d * d + mtp_block)
        return {"total": total, "activated": total - inactive, "mtp": mtp}

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "ModelConfig":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict] = {
    # 3.0M params (2.6M activated) with the 8K BPE vocab; ~1h per 10M tokens on a laptop CPU
    "tiny": dict(
        dim=128, n_layers=4, n_dense_layers=1, dense_hidden_dim=352,
        n_heads=4, kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, moe_hidden_dim=64,
        max_seq_len=1024,
    ),
    # 11.0M params (6.3M activated) with the 8K BPE vocab. Tuned for loss/usability over speed: deep-and-thin
    # (8 layers at dim 192), wide 128-dim experts, vocab tables under a third of params.
    "mini": dict(
        dim=192, n_layers=8, n_dense_layers=1, dense_hidden_dim=512,
        n_heads=6, kv_lora_rank=48, qk_nope_head_dim=24, qk_rope_head_dim=16, v_head_dim=24,
        n_routed_experts=12, n_shared_experts=1, n_activated_experts=3, moe_hidden_dim=128,
        n_expert_groups=3, n_limited_groups=2, max_seq_len=1024,
    ),
    # 14.3M params (8.4M activated) with the 8K BPE vocab
    "small": dict(
        dim=256, n_layers=6, n_dense_layers=1, dense_hidden_dim=704,
        n_heads=8, q_lora_rank=128, kv_lora_rank=64,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_routed_experts=16, n_shared_experts=1, n_activated_experts=4, moe_hidden_dim=128,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=1024,
    ),
    # 85.9M params (30.8M activated) with the 8K BPE vocab; a GPU is recommended
    "medium": dict(
        dim=512, n_layers=8, n_dense_layers=1, dense_hidden_dim=1408,
        n_heads=8, q_lora_rank=256, kv_lora_rank=128,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        n_routed_experts=24, n_shared_experts=1, n_activated_experts=4, moe_hidden_dim=256,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=2048,
    ),
    # 283.9M params (130.5M activated) with the 32K BPE vocab; wants a GPU
    "base": dict(
        dim=768, n_layers=12, n_dense_layers=2, dense_hidden_dim=2048,
        n_heads=12, q_lora_rank=384, kv_lora_rank=256,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        n_routed_experts=32, n_shared_experts=2, n_activated_experts=6, moe_hidden_dim=256,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=4096, init_std=0.006,
    ),
    # ---- Reference configs, far too big for a laptop. Sizes are from param_estimate(). ----
    # 1.31B params (0.32B activated) with a 32K vocab.
    "large": dict(
        vocab_size=32768, dim=1024, n_layers=16, n_dense_layers=1, dense_hidden_dim=2816,
        n_heads=16, q_lora_rank=512, kv_lora_rank=256,
        qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
        n_routed_experts=48, n_shared_experts=2, n_activated_experts=6, moe_hidden_dim=512,
        n_expert_groups=4, n_limited_groups=2, max_seq_len=4096, init_std=0.006,
    ),
    # 5.83B params (1.11B activated) with a 64K vocab; V2-style head dims.
    "xl": dict(
        vocab_size=65536, dim=1536, n_layers=24, n_dense_layers=1, dense_hidden_dim=4096,
        n_heads=16, q_lora_rank=768, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed_experts=64, n_shared_experts=2, n_activated_experts=6, moe_hidden_dim=768,
        n_expert_groups=8, n_limited_groups=4, max_seq_len=4096, init_std=0.006,
    ),
    # DeepSeek-V2-Lite shape: 15.7B params (2.4B activated excluding the embedding lookup).
    # The original used softmax gating with auxiliary losses; this uses V3-style routing.
    "v2-lite": dict(
        vocab_size=102400, dim=2048, n_layers=27, n_dense_layers=1, dense_hidden_dim=10944,
        n_heads=16, q_lora_rank=0, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed_experts=64, n_shared_experts=2, n_activated_experts=6, moe_hidden_dim=1408,
        max_seq_len=4096, init_std=0.006,
    ),
    # DeepSeek-V3: 671B params (37B activated), +11.6B MTP (the paper's 14B also counts
    # the MTP module's stored copy of the embedding and head). Pretrained at 4K, FP8.
    "v3": dict(
        vocab_size=129280, dim=7168, n_layers=61, n_dense_layers=3, dense_hidden_dim=18432,
        n_heads=128, q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        n_routed_experts=256, n_shared_experts=1, n_activated_experts=8, moe_hidden_dim=2048,
        n_expert_groups=8, n_limited_groups=4, route_scale=2.5,
        max_seq_len=4096, init_std=0.006, use_fp8=True,
    ),
}


# ---- Sparse twins: same compute per token, about 2x the total parameters ----
# Built from each dense preset with one rule, following fine-grained MoE practice
# (DeepSeekMoE, Qwen3-Next / AliceAI 80B-A3B): 5x more routed experts at half the width,
# 2x as many active per token (so active expert FLOPs match), 2x shared experts at half
# width (so the always-on path keeps its size), and V3-style 8-group routing.
# Sizes with the default BPE vocab (total / activated):
#   tiny-sparse   3.6M /   2.6M     mini-sparse   21.4M /   6.4M    small-sparse  26.2M / 8.5M
#   medium-sparse 185.3M / 31.2M    base-sparse  568.0M / 131.5M
# Note: equal FLOPs is not equal speed. Many small experts cost ~1.8x per MoE layer on CPU
# with the per-expert loop; a grouped-GEMM expert kernel is needed to close that gap on GPU.
SPARSE_OVERRIDES: dict[str, dict] = {
    "tiny":   dict(n_routed_experts=32,  moe_hidden_dim=32,  n_activated_experts=4,  n_shared_experts=2,
                   n_expert_groups=4, n_limited_groups=2),
    "mini":   dict(n_routed_experts=64,  moe_hidden_dim=64,  n_activated_experts=6,  n_shared_experts=2,
                   n_expert_groups=8, n_limited_groups=4),
    "small":  dict(n_routed_experts=80,  moe_hidden_dim=64,  n_activated_experts=8,  n_shared_experts=2,
                   n_expert_groups=8, n_limited_groups=4),
    "medium": dict(n_routed_experts=120, moe_hidden_dim=128, n_activated_experts=8,  n_shared_experts=2,
                   n_expert_groups=8, n_limited_groups=4),
    "base":   dict(n_routed_experts=160, moe_hidden_dim=128, n_activated_experts=12, n_shared_experts=4,
                   n_expert_groups=8, n_limited_groups=4),
}
for _name, _over in SPARSE_OVERRIDES.items():
    PRESETS[f"{_name}-sparse"] = {**PRESETS[_name], **_over}
