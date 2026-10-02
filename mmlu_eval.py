"""MMLU (cais/mmlu, 14,042 test questions, 57 subjects) for small-context models.

Two scoring formats:
  letter  standard MMLU: "Q / A. .. B. .. C. .. D. .. / Answer:" -> compare p(" A"), p(" B"), p(" C"), p(" D").
          Few-shot examples from the dev split (same subject) are added only while they fit.
  cloze   "Q / Answer:" + each choice's text, scored by log-prob per byte (length-normalised).
          Gives a signal earlier than the letter format for small models (as in SmolLM / OLMo evals).

Only questions that fit the model's context window are scored; anything that can't fit even
0-shot is skipped and counted, so coverage is reported alongside accuracy.

    python mmlu_eval.py                                   # checkpoints/mini.pt, full test set, both formats
    python mmlu_eval.py --limit-per-subject 20            # quick run (~1,140 questions)
    python mmlu_eval.py --ckpt checkpoints/mini-sft.pt --log mmlu_results.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.runtime import pick_device
from deepseek_moe.tokenizer import load_tokenizer

LETTERS = ["A", "B", "C", "D"]

# Standard MMLU category grouping (as in lm-evaluation-harness): 57 subjects -> 4 categories.
CATEGORIES = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry",
             "college_computer_science", "college_mathematics", "college_physics", "computer_security",
             "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology",
             "high_school_chemistry", "high_school_computer_science", "high_school_mathematics",
             "high_school_physics", "high_school_statistics", "machine_learning"],
    "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history",
                   "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies",
                   "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law",
                   "world_religions"],
    "Social Sciences": ["econometrics", "high_school_geography", "high_school_government_and_politics",
                        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
                        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
                        "sociology", "us_foreign_policy"],
    "Other": ["business_ethics", "clinical_knowledge", "college_medicine", "global_facts", "human_aging",
              "management", "marketing", "medical_genetics", "miscellaneous", "nutrition",
              "professional_accounting", "professional_medicine", "virology"],
}
CATEGORY_OF = {s: c for c, subjects in CATEGORIES.items() for s in subjects}

# A topical group (not an official MMLU category): the life-science and health subjects,
# where FineWeb-Edu is richest.
BIO_MED = ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "high_school_biology",
           "human_aging", "medical_genetics", "nutrition", "professional_medicine", "virology"]


def category_breakdown(correct: dict[str, float], counts: dict[str, int]) -> dict[str, dict]:
    """Question-weighted accuracy per category from per-subject correct counts and question counts."""
    out = {}
    for cat, subjects in CATEGORIES.items():
        n = sum(counts.get(s, 0) for s in subjects)
        c = sum(correct.get(s, 0) for s in subjects)
        out[cat] = {"accuracy": c / n if n else float("nan"), "questions": n, "subjects": len(subjects)}
    return out


def print_categories(fmt: str, cats: dict[str, dict]) -> None:
    print(f"  {fmt} by category:   " + "   ".join(
        f"{c} {v['accuracy']:.1%} ({v['questions']:,}q)" for c, v in cats.items()))


def load_split(split: str) -> list[dict]:
    path = hf_hub_download("cais/mmlu", f"all/{split}-00000-of-00001.parquet", repo_type="dataset")
    return pq.read_table(path).to_pylist()


def fmt_letter(q: dict, with_answer: bool) -> str:
    s = q["question"].strip() + "\n" + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, q["choices"])) + "Answer:"
    return s + (f" {LETTERS[q['answer']]}\n\n" if with_answer else "")


def fmt_cloze(q: dict, with_answer: bool) -> str:
    s = q["question"].strip() + "\nAnswer:"
    return s + (f" {q['choices'][q['answer']]}\n\n" if with_answer else "")


class MMLU:
    def __init__(self, ckpt: str, device: str):
        self.device = device
        ck = torch.load(ckpt, map_location=device)
        self.tok = load_tokenizer(**ck["tokenizer"])
        self.model = DeepSeekMoEModel(ModelConfig.from_dict(ck["config"])).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.max_len = self.model.cfg.max_seq_len
        self.step, self.chat = ck.get("step"), bool(ck.get("chat"))
        self.bos = [self.tok.eot_id]
        self.letter_ids = [self.tok.encode(f" {L}")[0] for L in LETTERS]

    def build(self, q: dict, shots: list[dict], k: int, fmt, reserve: int) -> tuple[list[int], int] | None:
        """(prompt ids, shots used) with as many of the first k shots as fit, leaving `reserve` tokens; or None."""
        header = f"The following are multiple choice questions (with answers) about {q['subject'].replace('_', ' ')}.\n\n"
        body = self.tok.encode(fmt(q, False))
        ids = self.bos + self.tok.encode(header)
        if len(ids) + len(body) + reserve > self.max_len:
            return None
        used = 0
        for shot in shots[:k]:
            s = self.tok.encode(fmt(shot, True))
            if len(ids) + len(s) + len(body) + reserve > self.max_len:
                break
            ids += s
            used += 1
        return ids + body, used

    @torch.no_grad()
    def letter(self, q: dict, shots: list[dict], k: int) -> tuple[bool, int] | None:
        built = self.build(q, shots, k, fmt_letter, reserve=1)
        if built is None:
            return None
        ids, used = built
        logits = self.model(torch.tensor([ids], device=self.device)).logits[0, -1]
        pred = int(logits[self.letter_ids].argmax())
        return pred == q["answer"], used

    @torch.no_grad()
    def cloze(self, q: dict, shots: list[dict], k: int) -> tuple[bool, int] | None:
        conts = [self.tok.encode(" " + str(c)) for c in q["choices"]]
        built = self.build(q, shots, k, fmt_cloze, reserve=max(len(c) for c in conts))
        if built is None:
            return None
        ids, used = built
        seqs = [ids + c for c in conts]
        T = max(len(s) for s in seqs)
        x = torch.full((4, T), self.tok.eot_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        logp = self.model(x[:, :-1].to(self.device)).logits.float().log_softmax(-1)
        scores = []
        for i, c in enumerate(conts):
            pos = torch.arange(len(ids) - 1, len(ids) - 1 + len(c))
            lp = logp[i, pos, torch.tensor(c)].sum().item()
            scores.append(lp / max(1, len((" " + str(q["choices"][i])).encode("utf-8"))))
        pred = max(range(4), key=lambda i: scores[i])
        return pred == q["answer"], used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/mini.pt")
    ap.add_argument("--formats", nargs="+", default=["letter", "cloze"], choices=["letter", "cloze"])
    ap.add_argument("--shots", type=int, default=5, help="max few-shot examples for the letter format (only those that fit)")
    ap.add_argument("--cloze-shots", type=int, default=0, help="max few-shot examples for the cloze format")
    ap.add_argument("--limit-per-subject", type=int, default=None, help="quick run: first N questions per subject")
    ap.add_argument("--log", default=None, help="append a JSON summary line here")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--summarize", default=None, metavar="RESULTS_JSONL",
                    help="print category breakdowns for runs already logged in this file, then exit")
    args = ap.parse_args()

    if args.summarize:
        test_counts = collections.Counter(q["subject"] for q in load_split("test"))
        for line in open(args.summarize):
            r = json.loads(line)
            print(f"{r['ckpt']} (step {r['step']}{', SFT' if r.get('chat') else ''}, {r['time']}, {r['questions']:,} questions)")
            for fmt in ("letter", "cloze"):
                if fmt not in r:
                    continue
                m = r[fmt]
                counts = m.get("subject_counts") or {s: test_counts[s] for s in m["by_subject"]}
                correct = {s: m["by_subject"][s] * counts[s] for s in m["by_subject"]}
                print(f"  {fmt}: overall {m['accuracy']:.1%}")
                print_categories(fmt, category_breakdown(correct, counts))
        return

    ev = MMLU(args.ckpt, pick_device(args.device))
    test, dev = load_split("test"), load_split("dev")
    shots_by_subject = collections.defaultdict(list)
    for d in dev:
        shots_by_subject[d["subject"]].append(d)
    if args.limit_per_subject:
        per, subset = collections.Counter(), []
        for q in test:
            per[q["subject"]] += 1
            if per[q["subject"]] <= args.limit_per_subject:
                subset.append(q)
        test = subset

    print(f"{args.ckpt} (step {ev.step}{', SFT' if ev.chat else ''}): {len(test):,} questions, "
          f"context {ev.max_len} tokens, formats {args.formats}", flush=True)
    summary = {"ckpt": args.ckpt, "step": ev.step, "chat": ev.chat, "time": time.strftime("%Y-%m-%d %H:%M"),
               "questions": len(test)}
    for fmt in args.formats:
        fn, k = (ev.letter, args.shots) if fmt == "letter" else (ev.cloze, args.cloze_shots)
        correct, scored, skipped, shots_used = collections.Counter(), collections.Counter(), 0, 0
        t0 = time.time()
        for i, q in enumerate(test, 1):
            r = fn(q, shots_by_subject[q["subject"]], k)
            if r is None:
                skipped += 1
            else:
                ok, n_shots = r
                scored[q["subject"]] += 1
                correct[q["subject"]] += ok
                shots_used += n_shots
            if i % 1000 == 0:
                done = sum(scored.values())
                print(f"  [{fmt}] {i:,}/{len(test):,}  acc {sum(correct.values()) / max(done, 1):.1%}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        n = sum(scored.values())
        acc = sum(correct.values()) / max(n, 1)
        macro = sum(correct[s] / scored[s] for s in scored) / max(len(scored), 1)
        print(f"\n{fmt}: accuracy {acc:.2%} (macro over subjects {macro:.2%}) on {n:,} scored questions; "
              f"skipped {skipped:,} that don't fit; avg shots used {shots_used / max(n, 1):.1f}  (chance = 25%)")
        best = sorted(scored, key=lambda s: -correct[s] / scored[s])
        print("  best subjects:  " + ", ".join(f"{s} {correct[s] / scored[s]:.0%}" for s in best[:5]))
        print("  worst subjects: " + ", ".join(f"{s} {correct[s] / scored[s]:.0%}" for s in best[-5:]))
        cats = category_breakdown(correct, scored)
        print_categories(fmt, cats)
        summary[fmt] = {"accuracy": acc, "macro_accuracy": macro, "scored": n, "skipped": skipped,
                        "avg_shots": shots_used / max(n, 1),
                        "by_subject": {s: correct[s] / scored[s] for s in sorted(scored)},
                        "subject_counts": {s: scored[s] for s in sorted(scored)},
                        "by_category": cats}
    if args.log:
        with open(args.log, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
        print(f"logged to {args.log}")


if __name__ == "__main__":
    main()
