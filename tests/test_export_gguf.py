import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_export_gguf_roundtrip(tmp_path):
    (tmp_path / "c.txt").write_text("The quick brown fox jumps over the lazy dog. " * 50)
    tok = BPETokenizer.train([str(tmp_path / "c.txt")], 300)
    cfg = ModelConfig.from_preset("mini", vocab_size=tok.vocab_size, n_layers=3, max_seq_len=64)
    torch.manual_seed(0)
    model = DeepSeekMoEModel(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 7,
                "tokenizer": {"kind": "bpe", "serialized": tok.to_str()}}, ckpt)

    r = subprocess.run([sys.executable, "export_gguf.py", "--ckpt", str(ckpt), "--dtype", "f32"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    g = gguf.GGUFReader(tmp_path / "m-f32.gguf")
    f = lambda k: g.fields[k].contents()
    assert f("general.architecture") == "deepseek2"
    assert f("deepseek2.block_count") == 3 and f("deepseek2.expert_count") == cfg.n_routed_experts
    assert f("deepseek2.attention.q_lora_rank") == 0  # required by llama.cpp for non-"lite" layer counts
    assert f("deepseek2.attention.key_length_mla") == cfg.qk_head_dim
    assert len(f("tokenizer.ggml.tokens")) == tok.vocab_size

    t = {x.name: np.asarray(x.data) for x in g.tensors}
    assert "blk.1.ffn_gate_exps.weight" in t and "blk.0.ffn_gate.weight" in t  # layer 0 dense, rest MoE
    # The k_b / v_b split must reassemble into the original KV up-projection exactly.
    H, nope, v, r_kv = cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    k_b = torch.from_numpy(t["blk.1.attn_k_b.weight"].reshape(H, r_kv, nope))
    v_b = torch.from_numpy(t["blk.1.attn_v_b.weight"].reshape(H, v, r_kv))
    rebuilt = torch.cat([k_b.transpose(1, 2), v_b], dim=1).reshape(H * (nope + v), r_kv)
    torch.testing.assert_close(rebuilt, model.layers[1].attn.wkv_b.weight.detach())
    experts = torch.from_numpy(t["blk.2.ffn_down_exps.weight"])
    torch.testing.assert_close(experts[5], model.layers[2].ffn.experts[5].w2.weight.detach())
