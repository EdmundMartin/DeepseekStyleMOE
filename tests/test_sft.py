import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.chat import END, IGNORE, add_chat_tokens, encode_conversation, render_prompt
from deepseek_moe.model import masked_cross_entropy
from deepseek_moe.tokenizer import BPETokenizer
from gen_sft_data import parse_pairs, split_documents

ROOT = Path(__file__).resolve().parents[1]
CORPUS = "Water boils at one hundred degrees Celsius at sea level. " * 40 + "Plants use sunlight to make food. " * 40


def chat_tok(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text(CORPUS)
    tok = BPETokenizer.train([str(f)], 300)
    add_chat_tokens(tok)
    return tok


def test_encode_conversation_masks_everything_but_replies(tmp_path):
    tok = chat_tok(tmp_path)
    msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Why?"},
            {"role": "assistant", "content": "Because."}]
    ids, labels = encode_conversation(tok, msgs, 100)
    trained = [i for i, l in zip(ids, labels) if l != IGNORE]
    assert tok.decode(trained[:-1]) == "Because." and trained[-1] == tok.token_to_id(END)
    # Prompt rendering matches the training layout up to the reply.
    assert render_prompt(tok, msgs[:2]) == ids[: len(render_prompt(tok, msgs[:2]))]


def test_encode_conversation_truncates_at_turn_boundary(tmp_path):
    tok = chat_tok(tmp_path)
    turn = [{"role": "user", "content": "Plants?"}, {"role": "assistant", "content": "Plants use sunlight."}]
    one = encode_conversation(tok, turn, 1000)[0]
    ids, _ = encode_conversation(tok, turn * 5, len(one) * 2 + 1)
    assert len(ids) == 2 * len(one)  # two whole exchanges kept, third dropped
    assert encode_conversation(tok, turn, len(one) - 1) is None


def test_resize_vocab_keeps_old_logits():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(ModelConfig.from_preset("tiny", max_seq_len=64)).eval()
    idx = torch.randint(0, 256, (1, 12))
    before = model(idx).logits
    model.resize_vocab(260)
    after = model(idx).logits
    assert after.shape[-1] == 260
    torch.testing.assert_close(after[..., :256], before)


def test_masked_cross_entropy_all_ignored_is_zero_not_nan():
    logits = torch.randn(2, 3, 10, requires_grad=True)
    loss = masked_cross_entropy(logits, torch.full((2, 3), -100))
    loss.backward()
    assert loss.item() == 0.0 and not torch.isnan(logits.grad).any()


def test_generate_stops_at_stop_token():
    torch.manual_seed(0)
    model = DeepSeekMoEModel(ModelConfig.from_preset("tiny", max_seq_len=64))
    # Greedy decoding is deterministic, so the first sampled token is known; use it as the stop token.
    first = model.generate(torch.zeros(1, 3, dtype=torch.long), 1, temperature=0.0)[0, -1].item()
    out = model.generate(torch.zeros(1, 3, dtype=torch.long), 20, temperature=0.0, stop_ids={first})
    assert out.shape[1] == 4


def test_parse_pairs_and_split_documents():
    reply = '<think>hmm</think>```json\n{"pairs": [{"question": "Why is the sky blue?", "answer": "Rayleigh scattering."},' \
            '{"question": "What does the document say?", "answer": "x"}]}\n```'
    assert parse_pairs(reply) == [{"question": "Why is the sky blue?", "answer": "Rayleigh scattering."}]
    assert parse_pairs("not json") == []
    data = np.array([1, 2, 9, 3, 9, 4, 5], dtype=np.uint16)
    assert split_documents(data, 9) == [(0, 2), (3, 4), (5, 7)]


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "fake-teacher"
        content = json.dumps({"pairs": [{"question": "At what temperature does water boil at sea level?",
                                         "answer": "Water boils at 100 degrees Celsius at sea level."}]})
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": "fake-teacher",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}]}
        raw = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


def test_generate_then_sft_end_to_end(tmp_path):
    # A tiny "pretraining" dataset and checkpoint, laid out like prepare_data.py / train.py output.
    d = tmp_path / "data"
    d.mkdir()
    (tmp_path / "c.txt").write_text(CORPUS)
    tok = BPETokenizer.train([str(tmp_path / "c.txt")], 300)
    tok.save(d / "tokenizer.json")
    doc = tok.encode(CORPUS)
    np.array((doc + [tok.eot_id]) * 3, dtype=np.uint16).tofile(d / "train.bin")
    cfg = ModelConfig.from_preset("tiny", vocab_size=tok.vocab_size, max_seq_len=128)
    ckpt = tmp_path / "base.pt"
    torch.save({"config": cfg.to_dict(), "model": DeepSeekMoEModel(cfg).state_dict(),
                "tokenizer": {"kind": "bpe", "serialized": tok.to_str()}}, ckpt)

    server = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        gen = subprocess.run([sys.executable, "gen_sft_data.py", "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                              "--model", "fake-teacher", "--data-dir", str(d), "--docs", "3", "--concurrency", "2"],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
    finally:
        server.shutdown()
    assert gen.returncode == 0, gen.stderr
    rows = [json.loads(l) for l in (d / "sft_qa.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["messages"][1]["role"] == "assistant"

    sft = subprocess.run([sys.executable, "sft.py", "--init", str(ckpt), "--data", str(d / "sft_qa.jsonl"),
                          "--epochs", "30", "--batch-size", "2", "--eval-every", "1000", "--log-every", "1000",
                          "--val-frac", "0.3", "--lr", "3e-3", "--warmup", "5"],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert sft.returncode == 0, sft.stderr
    out = torch.load(tmp_path / "base-sft.pt")
    assert out["chat"] and out["config"]["vocab_size"] == tok.vocab_size + 4
    losses = [float(l.split()[-1]) for l in sft.stdout.splitlines() if "val lm_loss" in l]
    assert losses[-1] < losses[0] / 2, sft.stdout  # it learned the answers
