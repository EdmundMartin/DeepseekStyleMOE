import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.fp8 import FP8Linear, dequantize_blockwise, quantize_blockwise
from deepseek_moe.mla import MLA


def tiny(**kw):
    return ModelConfig.from_preset("tiny", **kw)


def test_forward_and_losses():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(tiny(n_mtp_modules=2)).train()
    idx = torch.randint(0, 256, (2, 32))
    out = model(idx[:, :-1], idx[:, 1:])
    assert out.logits.shape == (2, 31, 256)
    assert {"lm_loss", "mtp_loss", "aux_loss"} <= out.metrics.keys()
    out.loss.backward()
    assert model.mtp[1].proj.weight.grad is not None


def test_mla_absorbed_matches_naive():
    torch.manual_seed(0)
    cfg = tiny(q_lora_rank=16)
    attn = MLA(cfg).eval()
    x = torch.randn(2, 10, cfg.dim)
    model = DeepSeekMoEModel(cfg)
    cos, sin = model.rope_cos[:10], model.rope_sin[:10]
    naive = attn(x, cos, sin)
    absorbed = attn(x, cos, sin, cache={})
    torch.testing.assert_close(naive, absorbed, atol=1e-5, rtol=1e-4)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(tiny()).eval()
    idx = torch.randint(0, 256, (1, 20))
    full = model(idx).logits[:, -1]
    caches = [{} for _ in model.layers]
    model.forward_cached(idx[:, :12], caches, 0)
    for t in range(12, 20):
        last = model.forward_cached(idx[:, t : t + 1], caches, t)
    torch.testing.assert_close(full, last, atol=1e-4, rtol=1e-4)
    # The cache stores only the latent + rope key, not per-head K/V.
    cfg = model.cfg
    assert caches[0]["c_kv"].shape == (1, 20, cfg.kv_lora_rank)
    assert caches[0]["k_pe"].shape == (1, 20, cfg.qk_rope_head_dim)


def test_generate():
    model = DeepSeekMoEModel(tiny(max_seq_len=16))
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=30, top_k=10)
    assert out.shape == (1, 34)


def test_routing_weights_and_group_limit():
    torch.manual_seed(0)
    cfg = tiny(n_routed_experts=8, n_activated_experts=2, n_expert_groups=4, n_limited_groups=1)
    gate = DeepSeekMoEModel(cfg).layers[-1].ffn.gate
    w, idx, _ = gate(torch.randn(64, cfg.dim))
    torch.testing.assert_close(w.sum(-1), torch.ones(64))
    # With one allowed group of 2 experts, both picks must share a group.
    assert ((idx // 2)[:, 0] == (idx // 2)[:, 1]).all()


def test_bias_update_pushes_toward_balance():
    cfg = tiny()
    moe = DeepSeekMoEModel(cfg).layers[-1].ffn
    moe.expert_load.copy_(torch.tensor([100.0, 0, 0, 0, 0, 0, 0, 0]))
    moe.update_bias()
    assert moe.gate.bias[0] < 0 and (moe.gate.bias[1:] > 0).all()
    assert moe.expert_load.sum() == 0


def test_fp8_quantization():
    torch.manual_seed(0)
    w = torch.randn(200, 300)
    q, s = quantize_blockwise(w, 128)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (2, 3)
    rel = (dequantize_blockwise(q, s, 128) - w).norm() / w.norm()
    assert rel < 0.05

    model = DeepSeekMoEModel(tiny(use_fp8=True)).eval()
    idx = torch.randint(0, 256, (1, 16))
    before = model(idx).logits
    model.freeze_fp8()
    fp8 = [m for m in model.modules() if isinstance(m, FP8Linear)]
    assert fp8 and all(m.weight.dtype == torch.float8_e4m3fn for m in fp8)
    torch.testing.assert_close(model(idx).logits, before, atol=1e-4, rtol=1e-4)


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(tiny()).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 256, (4, 33))
    for _ in range(60):
        loss = model(idx[:, :-1], idx[:, 1:]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_moe_biases()
    assert model(idx[:, :-1], idx[:, 1:]).metrics["lm_loss"] < 1.0


def test_param_estimate_matches_model_and_deepseek_papers():
    for name in ["tiny", "mini", "small"]:
        cfg = ModelConfig.from_preset(name, vocab_size=8192)
        assert cfg.param_estimate() == DeepSeekMoEModel(cfg).param_counts()
    v3 = ModelConfig.from_preset("v3").param_estimate()
    assert round(v3["total"] / 1e9) == 671 and round(v3["activated"] / 1e9) == 38  # paper: 671B / 37B
    lite = ModelConfig.from_preset("v2-lite").param_estimate()
    assert round(lite["total"] / 1e9, 1) == 15.7
