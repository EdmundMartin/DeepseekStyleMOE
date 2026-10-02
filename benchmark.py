"""A crude factual-knowledge benchmark for comparing checkpoints.

Two kinds of test, both scored on raw text (a leading <|endoftext|> marks the start of a document):
  * probes: prompt -> expected answer. Reports the log-probability of the whole answer, the rank of
    its first token among the whole vocab, and top-1 / top-5 hit rates.
  * pairs:  a true sentence vs a false one. The pair is won if the true one has lower bits/byte
    (length-normalised, so pairs with different token counts compare fairly).

    python benchmark.py                                         # checkpoints/mini.pt
    python benchmark.py --ckpt checkpoints/mini.pt checkpoints/mini-sft.pt   # side by side
    python benchmark.py --verbose --log benchmarks.jsonl        # per-item detail, append results + rebuild
                                                                # benchmark_report.html (open it in a browser)
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.runtime import pick_device
from deepseek_moe.tokenizer import ByteTokenizer, load_tokenizer

# (prompt, expected answer). Answers include their leading space, as they'd appear in text.
PROBES: list[tuple[str, str]] = [
    ("The Earth revolves around the", " Sun"),
    ("Water is made of hydrogen and", " oxygen"),
    ("The heart pumps", " blood"),
    ("Plants make their food through a process called", " photosynthesis"),
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("World War II ended in", " 1945"),
    ("HIV is a", " virus"),
    ("The American flag is red, white and", " blue"),
    ("The sun rises in the", " east"),
    ("Fish breathe using their", " gills"),
    ("2 + 2 =", " 4"),
    ("3 + 5 =", " 8"),
    ("7 + 6 =", " 13"),
]

# (true sentence, false sentence)
PAIRS: list[tuple[str, str]] = [
    ("HIV is a virus that attacks the immune system.", "HIV is a bacteria that attacks the immune system."),
    ("HIV is a virus that attacks the immune system.", "HIV is a mineral that attacks the immune system."),
    ("HIV is a virus that attacks the immune system.", "HIV is a virus that attacks the digestive system."),
    ("The Earth revolves around the Sun.", "The Sun revolves around the Earth."),
    ("Paris is the capital of France.", "Paris is the capital of Germany."),
    ("Water freezes at zero degrees Celsius.", "Water freezes at fifty degrees Celsius."),
    ("Plants absorb carbon dioxide and release oxygen.", "Plants absorb oxygen and release carbon dioxide."),
    ("The sun rises in the east and sets in the west.", "The sun rises in the west and sets in the east."),
    ("Fish breathe through gills.", "Fish breathe through lungs."),
    ("Ice is frozen water.", "Ice is frozen sand."),
    ("A week has seven days.", "A week has twelve days."),
    ("Humans have two lungs.", "Humans have six lungs."),
]


class Scorer:
    def __init__(self, ckpt_path: str, device: str):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location=device)
        info = ckpt.get("tokenizer")
        self.tok = load_tokenizer(**info) if info else ByteTokenizer()
        self.model = DeepSeekMoEModel(ModelConfig.from_dict(ckpt["config"])).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.step = ckpt.get("step")
        self.chat = bool(ckpt.get("chat"))
        self.bos = [self.tok.eot_id] if self.tok.eot_id is not None else []

    @torch.no_grad()
    def token_logprobs(self, ids: list[int]) -> torch.Tensor:
        """log p(ids[i] | ids[:i]) for i >= 1."""
        x = torch.tensor([ids], device=self.device)
        logp = self.model(x[:, :-1]).logits[0].float().log_softmax(-1)
        return logp.gather(1, x[0, 1:, None])[:, 0]

    @torch.no_grad()
    def probe(self, prompt: str, answer: str) -> dict:
        p_ids = self.bos + self.tok.encode(prompt)
        full = self.bos + self.tok.encode(prompt + answer)
        if full[: len(p_ids)] != p_ids:  # tokenisation merged across the boundary; score the answer on its own
            full = p_ids + self.tok.encode(answer)
        a_ids = full[len(p_ids):]
        lp = self.token_logprobs(full)[len(p_ids) - 1:]
        with torch.no_grad():
            logits = self.model(torch.tensor([p_ids], device=self.device)).logits[0, -1].float()
        rank = int((logits > logits[a_ids[0]]).sum()) + 1
        top = [self.tok.decode([i]) for i in logits.topk(3).indices.tolist()]
        return {"prompt": prompt, "answer": answer, "logprob": float(lp.sum()), "p_first": float(lp[0].exp()),
                "rank": rank, "top3": top}

    def bits_per_byte(self, text: str) -> float:
        ids = self.bos + self.tok.encode(text)
        return float(-self.token_logprobs(ids).sum()) / math.log(2) / len(text.encode("utf-8"))

    def pair(self, true: str, false: str) -> dict:
        t, f = self.bits_per_byte(true), self.bits_per_byte(false)
        return {"true": true, "false": false, "bpb_true": t, "bpb_false": f, "won": t < f}

    def run(self) -> dict:
        probes = [self.probe(p, a) for p, a in PROBES]
        pairs = [self.pair(t, f) for t, f in PAIRS]
        return {
            "step": self.step, "chat": self.chat, "probes": probes, "pairs": pairs,
            "top1": sum(p["rank"] == 1 for p in probes) / len(probes),
            "top5": sum(p["rank"] <= 5 for p in probes) / len(probes),
            "mean_answer_logprob": sum(p["logprob"] for p in probes) / len(probes),
            "pair_acc": sum(p["won"] for p in pairs) / len(pairs),
            "mean_pair_margin_bpb": sum(p["bpb_false"] - p["bpb_true"] for p in pairs) / len(pairs),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", default=["checkpoints/mini.pt"])
    ap.add_argument("--verbose", action="store_true", help="print every probe and pair")
    ap.add_argument("--log", default=None, help="append results as JSON lines to this file")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = pick_device(args.device)

    results = {}
    for path in args.ckpt:
        r = Scorer(path, device).run()
        results[path] = r
        if args.verbose:
            print(f"\n=== {path} (step {r['step']}{', chat' if r['chat'] else ''})")
            print(f"  {'probe':52s} {'answer':16s} {'rank':>6s} {'p(1st)':>7s} {'logp':>7s}  top-3")
            for p in r["probes"]:
                mark = "✓" if p["rank"] == 1 else ("~" if p["rank"] <= 5 else "✗")
                print(f"{mark} {p['prompt'][:52]:52s} {p['answer']!r:16s} {p['rank']:6d} {p['p_first']:7.1%} "
                      f"{p['logprob']:7.2f}  {p['top3']}")
            print(f"  {'true vs false':80s} {'bpb T':>6s} {'bpb F':>6s}")
            for p in r["pairs"]:
                print(f"{'✓' if p['won'] else '✗'} {p['true'][:38]:38s} | {p['false'][:39]:39s} "
                      f"{p['bpb_true']:6.3f} {p['bpb_false']:6.3f}")
        if args.log:
            with open(args.log, "a") as fh:
                fh.write(json.dumps({"ckpt": path, "time": time.strftime("%Y-%m-%d %H:%M"), **r}) + "\n")

    print(f"\n{'checkpoint':40s} {'step':>6s} {'probe top-1':>11s} {'top-5':>6s} {'answer logp':>11s} "
          f"{'pairs won':>9s} {'margin':>7s}")
    for path, r in results.items():
        print(f"{path[-40:]:40s} {str(r['step']):>6s} {r['top1']:11.0%} {r['top5']:6.0%} "
              f"{r['mean_answer_logprob']:11.2f} {r['pair_acc']:9.0%} {r['mean_pair_margin_bpb']:+7.3f}")
    print(f"({len(PROBES)} probes, {len(PAIRS)} pairs; margin = mean bits/byte by which the true sentence wins)")
    if args.log:
        import subprocess
        import sys
        subprocess.run([sys.executable, "benchmark_report.py", "--log", args.log], check=False)


if __name__ == "__main__":
    main()
