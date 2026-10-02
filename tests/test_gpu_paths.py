"""GPU-oriented code paths, exercised on CPU so they stay correct without a GPU."""
import subprocess
import sys
from pathlib import Path

import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.runtime import autocast, pick_device, resolve_precision

ROOT = Path(__file__).resolve().parents[1]


def test_sorted_dispatch_matches_loop_forward_and_backward():
    torch.manual_seed(0)
    cfg = ModelConfig.from_preset("mini", vocab_size=512, n_layers=2)
    moe = DeepSeekMoEModel(cfg).layers[1].ffn
    x = torch.randn(3 * 40, cfg.dim)
    w, i, _ = moe.gate(x)
    c = torch.bincount(i.flatten(), minlength=cfg.n_routed_experts)
    grads = []
    for fn in (moe._dispatch_loop, moe._dispatch_sorted):
        moe.zero_grad()
        xx = x.clone().requires_grad_(True)
        out = fn(xx, w.detach(), i, c)
        out.pow(2).sum().backward()
        grads.append((out.detach(), xx.grad, moe.experts[3].w2.weight.grad.clone()))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b)


def test_grad_accumulation_matches_full_batch():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(ModelConfig.from_preset("tiny", vocab_size=256, max_seq_len=32)).train()
    x = torch.randint(0, 256, (4, 33))

    model.zero_grad()
    model(x[:, :-1], x[:, 1:]).loss.backward()
    full = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    model.zero_grad()
    for half in (x[:2], x[2:]):
        (model(half[:, :-1], half[:, 1:]).loss / 2).backward()
    accum = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    for a, b in zip(full, accum):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)


def test_router_stays_fp32_under_autocast():
    model = DeepSeekMoEModel(ModelConfig.from_preset("tiny", vocab_size=256, max_seq_len=32))
    gate = model.layers[-1].ffn.gate
    with autocast("cpu", "bf16"):
        w, _, scores = gate(torch.randn(8, model.cfg.dim))
    assert scores.dtype == torch.float32


def test_runtime_defaults_keep_cpu_exact():
    assert resolve_precision("auto", "cpu") == "fp32"
    assert pick_device("cpu") == "cpu"


def test_train_script_bf16_grad_accum_smoke(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("The quick brown fox jumps over the lazy dog. " * 400)
    r = subprocess.run([sys.executable, "train.py", "--data", str(corpus), "--tokenizer", "bytes", "--preset", "tiny",
                        "--set", "max_seq_len=64", "--steps", "4", "--batch-size", "2", "--grad-accum", "2",
                        "--precision", "bf16", "--eval-every", "2", "--eval-iters", "1", "--log-every", "1",
                        "--out", str(tmp_path / "m.pt")], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "grad-accum 2" in r.stdout and "precision bf16" in r.stdout and "tok/s" in r.stdout
    assert torch.load(tmp_path / "m.pt")["step"] == 3


def test_gather_rows_gradcheck_and_dispatch_choice():
    from deepseek_moe.moe import SORTED_DISPATCH_MIN_EXPERTS_CPU, _GatherRows
    x = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    idx = torch.tensor([0, 3, 3, 6, 1, 0])
    assert torch.autograd.gradcheck(lambda t: _GatherRows.apply(t, idx), (x,))
    # Sparse presets (many experts) take the sorted path on CPU; mini/small keep the loop.
    assert ModelConfig.from_preset("small-sparse").n_routed_experts >= SORTED_DISPATCH_MIN_EXPERTS_CPU
    assert ModelConfig.from_preset("mini").n_routed_experts < SORTED_DISPATCH_MIN_EXPERTS_CPU


def test_sparse_model_trains_identically_via_either_dispatch():
    torch.manual_seed(0)
    cfg = ModelConfig.from_preset("mini-sparse", vocab_size=256, n_layers=2, max_seq_len=32)
    model = DeepSeekMoEModel(cfg).train()
    x = torch.randint(0, 256, (2, 17))
    grads = []
    for force_loop in (True, False):
        import deepseek_moe.moe as moe_mod
        old = moe_mod.SORTED_DISPATCH_MIN_EXPERTS_CPU
        moe_mod.SORTED_DISPATCH_MIN_EXPERTS_CPU = 10**9 if force_loop else 1
        try:
            model.zero_grad()
            out = model(x[:, :-1], x[:, 1:])
            out.loss.backward()
            grads.append((out.loss.item(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
        finally:
            moe_mod.SORTED_DISPATCH_MIN_EXPERTS_CPU = old
        model.update_moe_biases(update=False)
    assert abs(grads[0][0] - grads[1][0]) < 1e-6
    for a, b in zip(grads[0][1], grads[1][1]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
