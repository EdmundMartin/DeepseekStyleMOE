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


def test_sample_next_top_p_and_repetition_penalty():
    from deepseek_moe.model import sample_next

    logits = torch.tensor([[4.0, 3.0, 1.0, -2.0]])
    ctx = torch.tensor([[0]])
    # Greedy picks 0; a strong repetition penalty on token 0 flips it to 1.
    assert sample_next(logits, ctx, temperature=0).item() == 0
    assert sample_next(logits, ctx, temperature=0, repetition_penalty=2.0).item() == 1
    # Negative logits are pushed further down, never up.
    assert sample_next(torch.tensor([[-1.0, -1.5]]), torch.tensor([[0]]), temperature=0, repetition_penalty=2.0).item() == 1
    # Tiny top_p keeps only the most likely token, so sampling becomes deterministic.
    torch.manual_seed(0)
    assert all(sample_next(logits, ctx, temperature=1.0, top_p=0.01).item() == 0 for _ in range(50))
    # top_p=0.9 here keeps tokens 0 and 1 (p ~0.71, ~0.26) and drops the tail.
    picks = {sample_next(logits, ctx, temperature=1.0, top_p=0.9).item() for _ in range(300)}
    assert picks == {0, 1}


def test_generate_with_sampling_options():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(tiny(max_seq_len=32))
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), 20, temperature=0.7, top_k=20,
                         top_p=0.9, repetition_penalty=1.2)
    assert out.shape == (1, 24)


def test_sparse_twins_keep_compute_and_add_capacity():
    from deepseek_moe.config import PRESETS
    from deepseek_moe.tokenizer import DEFAULT_BPE_VOCAB

    for name in ["tiny", "mini", "small", "medium", "base"]:
        v = DEFAULT_BPE_VOCAB[f"{name}-sparse"]
        dense = ModelConfig.from_preset(name, vocab_size=v).param_estimate()
        sparse = ModelConfig.from_preset(f"{name}-sparse", vocab_size=v).param_estimate()
        assert abs(sparse["activated"] / dense["activated"] - 1) < 0.03, name   # same compute per token
        assert sparse["total"] > 1.15 * dense["total"], name                     # more capacity
        assert PRESETS[f"{name}-sparse"]["dim"] == PRESETS[name]["dim"]          # only the experts change
    # The sparse twin actually builds and runs.
    m = DeepSeekMoEModel(ModelConfig.from_preset("mini-sparse", vocab_size=512, n_layers=2, max_seq_len=32))
    x = torch.randint(0, 512, (2, 17))
    assert torch.isfinite(m(x[:, :-1], x[:, 1:]).loss)


def test_padding_mask_excludes_pads_from_moe_load_and_aux():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(tiny(vocab_size=256, max_seq_len=32)).train()
    x = torch.randint(1, 256, (2, 17))
    # Unmasked vs all-true mask: identical aux loss and load (pretraining behaviour unchanged).
    out_a = model(x[:, :-1], x[:, 1:])
    load_a = [m.expert_load.clone() for m in model.moe_layers()]
    model.update_moe_biases(update=False)
    out_b = model(x[:, :-1], x[:, 1:], token_mask=torch.ones(2, 16, dtype=torch.bool))
    load_b = [m.expert_load.clone() for m in model.moe_layers()]
    model.update_moe_biases(update=False)
    assert torch.isclose(out_a.loss, out_b.loss)
    assert all(torch.equal(a, b) for a, b in zip(load_a, load_b))
    # Masking the second half of each sequence removes those tokens from the counts.
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :8] = True
    model(x[:, :-1], x[:, 1:], token_mask=mask)
    k = model.cfg.n_activated_experts
    assert all(m.ffn.expert_load.sum().item() == 2 * 8 * k for m in model.layers if hasattr(m.ffn, "expert_load"))
    assert all(m.token_mask is None for m in model.moe_layers())  # mask is cleared after the forward


def test_frozen_bias_updates_report_but_do_not_change_biases():
    model = DeepSeekMoEModel(tiny(vocab_size=256, max_seq_len=32)).train()
    x = torch.randint(0, 256, (2, 17))
    model(x[:, :-1], x[:, 1:])
    before = [m.gate.bias.clone() for m in model.moe_layers()]
    stats = model.update_moe_biases(update=False)
    assert "max_load_ratio" in stats
    assert all(torch.equal(a, m.gate.bias) for a, m in zip(before, model.moe_layers()))
    assert all(m.expert_load.sum() == 0 for m in model.moe_layers())
