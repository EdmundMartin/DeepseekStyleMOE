"""Build a Hugging Face release folder (weights, GGUF, tokenizer, model card) from a checkpoint.

    python prepare_release.py --ckpt checkpoints/mini.pt --repo <user>/ShallowSeek-mini-base
    hf upload <user>/ShallowSeek-mini-base release/ShallowSeek-mini-base    # when you're happy
    python prepare_release.py --ckpt checkpoints/mini-sft.pt     # -> ShallowSeek-mini-chat

Metrics in the card come from the training log (validation loss), benchmarks.jsonl and
mmlu_results.jsonl (latest entries for this checkpoint's step), so re-run after evaluating.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from deepseek_moe import DeepSeekMoEModel, ModelConfig


def latest(path: str, step, ckpt: str | None = None) -> dict | None:
    """Most recent logged result for this checkpoint (matched on path when given, and on step)."""
    if not Path(path).exists():
        return None
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("step") == step and (ckpt is None or r.get("ckpt") == ckpt)]
    return rows[-1] if rows else None


OLLAMA_TEMPLATE = """{{- range .Messages }}
{{- if eq .Role "system" }}<|system|>{{ .Content }}<|end|>
{{- else if eq .Role "user" }}<|user|>{{ .Content }}<|end|>
{{- else if eq .Role "assistant" }}<|assistant|>{{ .Content }}<|end|>
{{- end }}
{{- end }}<|assistant|>"""
OLLAMA_PARAMS = {"stop": ["<|end|>", "<|user|>", "<|endoftext|>"], "temperature": 0.7, "top_p": 0.9,
                 "repeat_penalty": 1.15, "num_ctx": 1024}


def sft_stats(log: str) -> dict:
    """Pull data and loss figures from an sft.py log."""
    if not Path(log).exists():
        return {}
    text = Path(log).read_text()
    out = {}
    if m := re.search(r"([\d,]+) conversations -> ([\d,]+) usable \(([\d,]+) train / ([\d,]+) val\), ([\d.]+)M assistant", text):
        out.update(conversations=m.group(2), train=m.group(3), val=m.group(4), assistant_tokens_m=float(m.group(5)))
    if m := re.search(r"val lm_loss before SFT ([\d.]+)", text):
        out["val_before"] = float(m.group(1))
    vals = re.findall(r"^  val lm_loss ([\d.]+)", text, re.M)
    if vals:
        out["val_after"] = float(vals[-1])
    if m := re.findall(r"^step\s+\d+/(\d+)", text, re.M):
        out["steps"] = int(m[-1])
    return out


def card_it(args, cfg: ModelConfig, counts: dict, sft: dict, base_bench, base_mmlu, bench, mmlu, files) -> str:
    def pct(x):
        return f"{x:.1%}" if x is not None else "–"

    def row(label, b, i, fmt=pct):
        return f"| {label} | {fmt(b) if b is not None else '–'} | {fmt(i) if i is not None else '–'} |"

    rows = []
    if base_bench or bench:
        g = lambda r, k: r[k] if r else None
        rows += [row("Probes top-5", g(base_bench, "top5"), g(bench, "top5")),
                 row("Mean answer log-prob", g(base_bench, "mean_answer_logprob"), g(bench, "mean_answer_logprob"),
                     lambda x: f"{x:.2f}"),
                 row("True-vs-false pairs won", g(base_bench, "pair_acc"), g(bench, "pair_acc"))]
    for fmt in ("letter", "cloze"):
        b = base_mmlu[fmt]["accuracy"] if base_mmlu and fmt in base_mmlu else None
        i = mmlu[fmt]["accuracy"] if mmlu and fmt in mmlu else None
        if b is not None or i is not None:
            rows.append(row(f"MMLU ({fmt})", b, i))
    if mmlu and "cloze" in mmlu:
        from mmlu_eval import BIO_MED
        def biomed(r):
            if not r or "cloze" not in r:
                return None
            m = r["cloze"]
            counts_ = m.get("subject_counts")
            if not counts_:
                from mmlu_eval import load_split
                counts_ = {}
                for q in load_split("test"):
                    counts_[q["subject"]] = counts_.get(q["subject"], 0) + 1
            n = sum(counts_[s] for s in BIO_MED)
            return sum(m["by_subject"][s] * counts_[s] for s in BIO_MED) / n
        rows.append(row("MMLU biology & medicine (cloze)", biomed(base_mmlu), biomed(mmlu)))
    comparison = "| Metric | Base | **IT** |\n|---|---|---|\n" + "\n".join(rows) if rows else "*(no evaluations logged yet)*"
    loss = ""
    if "val_before" in sft:
        loss = (f"Validation loss on {sft.get('val', '?')} held-out conversations (assistant tokens only): "
                f"**{sft['val_before']:.3f} → {sft.get('val_after', float('nan')):.3f}**.")
    file_rows = "\n".join(f"| `{n}` | {d} |" for n, d in files)
    base_link = f"[{args.base_repo.split('/')[-1]}](https://huggingface.co/{args.base_repo})"
    code = f"[the training code]({args.code_url})" if args.code_url else "the training code"
    return f"""---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
base_model: {args.base_repo}
base_model_relation: finetune
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smol-smoltalk
tags:
- mixture-of-experts
- moe
- deepseek
- multi-head-latent-attention
- mla
- gguf
- from-scratch
- tiny
- chat
- instruction-tuned
---

# {args.name}

`{args.name}` is the **instruction-tuned (chat) version of {base_link}**, a tiny DeepSeek-V3-style Mixture-of-Experts model trained from scratch on a laptop CPU. It has **{counts['total'] / 1e6:.1f}M parameters**, of which **{counts['activated'] / 1e6:.1f}M are active per token**.

*Part of the ShallowSeek family. Not affiliated with DeepSeek.*

It answers in a chat format and stops when it's done, but it's still an **11M-parameter model**: answers are fluent and on-topic but often **wrong or vague**, and it **invents facts confidently**. It's an educational and research artifact; don't use it for anything that matters.

## Chat format

```
<|system|>You are a helpful assistant.<|end|><|user|>What causes the seasons?<|end|><|assistant|>
```

The model writes its reply and ends it with `<|end|>`. The GGUF embeds this template, and the GGUF repo includes Ollama `template` and `params` files.

## Fine-tuning (SFT)

| | |
|---|---|
| Base model | {base_link} (pretrained on 286M FineWeb-Edu tokens) |
| Conversations | {sft.get('conversations', '?')} ({sft.get('train', '?')} train / {sft.get('val', '?')} validation), {sft.get('assistant_tokens_m', '?')}M assistant tokens trained on |
| Grounded Q&A (~36% of assistant tokens) | ~10.6K single-turn question/answer pairs generated by **Qwen3.8-9B** from FineWeb-Edu documents **in ShallowSeek's own pretraining data**, so answers concern text the model saw. About 2% of pairs still refer to "the document"; these were left in |
| General chat (~64%) | 5,684 conversations from **HuggingFaceTB/smol-smoltalk**: all 1,994 everyday-conversations (small talk, multi-turn), plus openhermes, smol-constraints and smol-magpie-ultra-short (capped at 15% of assistant tokens), filtered to remove code and to fit 1,024 tokens |
| Loss | On assistant replies and their closing `<|end|>` only; prompts and padding are masked |
| Schedule | {sft.get('steps', '?')} steps, batch 16, 2 epochs, AdamW, LR 1.5e-4 → 1.5e-5 cosine (about 10% of the pretraining peak) |
| MoE routing | Load-balancing biases **frozen** at their pretrained values during SFT, with padding excluded from the expert-load statistics |

{loss}

## Evaluation: base vs IT

Both are scored on the same raw-text prompts, without the chat template, so the numbers are directly comparable. Chance on MMLU is 25%.

{comparison}

SFT mainly teaches format, turn-taking and when to stop, and MMLU is essentially unchanged. The factual probes improved slightly, plausibly because the grounded Q&A restates facts from documents the model pretrained on. With 14 probes and 12 pairs, treat that as a modest effect rather than a large one.

## Usage

**Ollama** (uses the bundled chat template):

```sh
ollama run hf.co/{args.repo or ('<user>/' + args.name)}-GGUF:Q8_0
```

**llama.cpp:**

```sh
llama-cli -m {args.name}-q8_0.gguf -cnv --temp 0.7 --top-p 0.9 --repeat-penalty 1.15
```

**PyTorch:** see `chat.py` in {code}:

```sh
python chat.py --ckpt mini-sft.pt --temperature 0.7 --rep-penalty 1.15
```

## Files

| File | Contents |
|---|---|
{file_rows}

## Acknowledgements

- Architecture after **DeepSeek-V3** (DeepSeek-AI, 2024).
- Pretraining data: **FineWeb-Edu** (Hugging Face, ODC-BY 1.0).
- Chat data: **smol-smoltalk** (Hugging Face, Apache-2.0).
- Synthetic Q&A written by **Qwen3.8-9B**.

*Card generated {time.strftime('%Y-%m-%d')} by `prepare_release.py`.*
"""


def mmlu_categories(m: dict) -> dict:
    """Per-category accuracy for one MMLU format result, from stored data or per-subject results."""
    if "by_category" in m:
        return m["by_category"]
    from mmlu_eval import category_breakdown, load_split
    test_counts = {}
    for q in load_split("test"):
        test_counts[q["subject"]] = test_counts.get(q["subject"], 0) + 1
    counts = m.get("subject_counts") or {s: test_counts[s] for s in m["by_subject"]}
    return category_breakdown({s: m["by_subject"][s] * counts[s] for s in m["by_subject"]}, counts)


def bio_med_section(m: dict) -> str:
    """Pooled Biology & Medicine cloze result plus per-subject rows, with significance vs chance."""
    from mmlu_eval import BIO_MED, load_split
    counts = m.get("subject_counts")
    if not counts:
        counts = {}
        for q in load_split("test"):
            counts[q["subject"]] = counts.get(q["subject"], 0) + 1
    subs = [s for s in BIO_MED if s in m["by_subject"]]
    n = sum(counts[s] for s in subs)
    acc = sum(m["by_subject"][s] * counts[s] for s in subs) / n
    z = (acc - 0.25) / math.sqrt(0.25 * 0.75 / n)
    rows = "\n".join(f"| {s.replace('_', ' ')} | {counts[s]} | {m['by_subject'][s]:.1%} |"
                     for s in sorted(subs, key=lambda s: -m["by_subject"][s]))
    above = sum(m["by_subject"][s] >= 0.25 for s in subs)
    return f"""
**Biology & medicine (cloze): {acc:.1%} on {n:,} questions**, against 25% chance. That's {(acc - 0.25) * 100:+.1f} points, about {z:.1f} standard errors above chance, so it's statistically significant. {above} of the {len(subs)} subjects are at or above chance. The overall cloze score is close to chance, so the signal sits mostly in the life-science and health subjects, where FineWeb-Edu's educational web text is richest. Maths-heavy subjects come out below chance, since cloze can't guess numeric answers. This group isn't an official MMLU category; it pools these {len(subs)} subjects:

| Subject | Questions | Cloze accuracy |
|---|---|---|
{rows}
"""


def val_curve(log: str) -> list[float]:
    return [float(x) for x in re.findall(r"val lm_loss ([\d.]+)", Path(log).read_text())] if Path(log).exists() else []


def card(args, cfg: ModelConfig, counts: dict, step, tok_meta: dict, vals: list[float], bench, mmlu, files) -> str:
    final = vals[-1] if vals else None
    bpt = tok_meta.get("bytes_per_token", 3.9)
    loss_lines = ""
    if final is not None:
        loss_lines = (f"| Validation loss (nats/token) | **{final:.3f}** |\n"
                      f"| Perplexity | {math.exp(final):.1f} |\n"
                      f"| Bits per byte | {final / bpt / math.log(2):.3f} |\n")
    curve = ""
    if vals:
        marks = [0, 4, 9, 14, 19, 24, 29, len(vals) - 1]
        curve = " → ".join(f"{vals[i]:.2f} (step {(i + 1) * 1000:,})" for i in sorted(set(m for m in marks if m < len(vals))))
    bench_md = ""
    if bench:
        probes = "\n".join(f"| {p['prompt']} | {p['answer'].strip()} | {p['rank']:,} | {', '.join(t.strip() or repr(t) for t in p['top3'])} |"
                           for p in bench["probes"])
        pairs_won = sum(p["won"] for p in bench["pairs"])
        bench_md = f"""
### Factual probes (`benchmark.py`)

These are 14 fill-in-the-blank probes and {len(bench['pairs'])} true-versus-false sentence pairs, scored on raw text.

| Metric | Value |
|---|---|
| Probes answered at top-1 / top-5 | {bench['top1']:.0%} / {bench['top5']:.0%} |
| Mean log-probability of the correct answer | {bench['mean_answer_logprob']:.2f} |
| True-vs-false pairs won | {pairs_won}/{len(bench['pairs'])} ({bench['pair_acc']:.0%}) |

| Prompt | Expected | Rank of expected | Model's top-3 |
|---|---|---|---|
{probes}
"""
    mmlu_md = ""
    if mmlu:
        rows = []
        for fmt in ("letter", "cloze"):
            if fmt in mmlu:
                m = mmlu[fmt]
                rows.append(f"| {fmt} | {m['accuracy']:.1%} | {m['scored']:,} (skipped {m['skipped']}) | {m['avg_shots']:.1f} |")
        fmts = [f for f in ("letter", "cloze") if f in mmlu]
        cats = {f: mmlu_categories(mmlu[f]) for f in fmts}
        cat_names = list(next(iter(cats.values())).keys()) if cats else []
        cat_rows = "\n".join(
            f"| {c} | {next(iter(cats.values()))[c]['subjects']} | {next(iter(cats.values()))[c]['questions']:,} | "
            + " | ".join(f"{cats[f][c]['accuracy']:.1%}" for f in fmts) + " |"
            for c in cat_names)
        mmlu_md = f"""
### MMLU (`mmlu_eval.py`)

| Format | Accuracy | Questions scored | Average few-shot examples |
|---|---|---|---|
{chr(10).join(rows)}

By category (accuracy weighted by question count; chance is 25%):

| Category | Subjects | Questions | {" | ".join(f.capitalize() for f in fmts)} |
|---|---|---|{"---|" * len(fmts)}
{cat_rows}

{bio_med_section(mmlu["cloze"]) if "cloze" in mmlu else ""}
Chance is 25%. In the **letter** format the model compares " A", " B", " C" and " D", with up to 5 same-subject examples included only while they fit in the 1,024-token window. In the **cloze** format each answer's text is scored by log-probability per byte. Small models in the letter format tend to answer the same letter within a subject, so per-subject letter scores mostly reflect how the answer key happens to be distributed. The cloze results are the meaningful ones here.
"""
    file_rows = "\n".join(f"| `{name}` | {desc} |" for name, desc in files)
    return f"""---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
tags:
- mixture-of-experts
- moe
- deepseek
- multi-head-latent-attention
- mla
- multi-token-prediction
- gguf
- from-scratch
- tiny
---

# {args.name}

`{args.name}` is a **tiny DeepSeek-V3-style Mixture-of-Experts language model trained from scratch on a laptop CPU**. It has **{counts['total'] / 1e6:.1f}M parameters** in total, of which **{counts['activated'] / 1e6:.1f}M are active per token**. It was pretrained on {tok_meta.get('train_tokens', 0) / 1e6:.0f}M tokens of FineWeb-Edu.

*Part of the ShallowSeek family. Not affiliated with DeepSeek. The name is a nod to the architecture it borrows, at a fraction of the depth.*

This is the **base model**, before any instruction tuning. It continues text; it doesn't follow instructions. It's an educational and research artifact: it writes fluent, on-topic English but **knows very few facts and confidently makes things up**. Don't use it for anything that matters.

## Architecture

It follows the main ideas of DeepSeek-V3, scaled down:

| Component | This model |
|---|---|
| Layers / hidden size | {cfg.n_layers} / {cfg.dim} (layer 0 has a dense FFN of {cfg.dense_hidden_dim}; layers 1–{cfg.n_layers - 1} are MoE) |
| Attention | **Multi-head Latent Attention (MLA)**: {cfg.n_heads} heads, KV latent {cfg.kv_lora_rank}, decoupled RoPE dim {cfg.qk_rope_head_dim}, head dims {cfg.qk_nope_head_dim}+{cfg.qk_rope_head_dim} (q/k) and {cfg.v_head_dim} (v). The KV cache stores only the {cfg.kv_lora_rank}-dim latent plus a {cfg.qk_rope_head_dim}-dim RoPE key per token |
| MoE | **DeepSeekMoE**: {cfg.n_routed_experts} fine-grained routed experts (SwiGLU, {cfg.moe_hidden_dim} hidden), top-{cfg.n_activated_experts}, plus {cfg.n_shared_experts} shared expert; sigmoid gating; group-limited routing ({cfg.n_expert_groups} groups, top-{cfg.n_limited_groups}) |
| Load balancing | **Auxiliary-loss-free** bias balancing (update speed {cfg.bias_update_speed}) plus a small sequence-wise auxiliary loss (α = {cfg.seq_aux_loss_alpha}). Expert load stayed at about 1.14× the mean (1.28× in the busiest layer) for most of training |
| Multi-Token Prediction | {cfg.n_mtp_modules} MTP module (λ = {cfg.mtp_loss_weight}), used in training only. It's in `model.safetensors` and left out of the GGUF ({counts['mtp'] / 1e6:.2f}M extra params) |
| Context | {cfg.max_seq_len} tokens |
| Tokenizer | Byte-level BPE, {cfg.vocab_size:,} tokens, trained on the same FineWeb-Edu slice (~{bpt:.1f} bytes/token) |

## Training

| | |
|---|---|
| Data | FineWeb-Edu `sample/10BT`, {tok_meta.get('documents', 0):,} documents, **{tok_meta.get('train_tokens', 0) / 1e6:.0f}M tokens** (~{tok_meta.get('train_tokens', 0) / counts['total']:.0f} tokens per parameter), one pass |
| Steps | {step + 1 if isinstance(step, int) else '?':,} × 8,192 tokens (batch 8 × 1,024) |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1, gradient clipping 1.0 |
| Learning rate | 1.5e-3 peak, 700 warm-up steps, cosine decay to 1.5e-4 |
| Precision | fp32 |
| Hardware | One 8-core Intel Core i9-9880H (2019 MacBook Pro), CPU only: about 2 days of compute at ~1.9K tokens/s |

Validation loss: {curve}

## Evaluation

| Metric | Value |
|---|---|
{loss_lines}{bench_md}{mmlu_md}
## What it can and can't do

- ✅ Grammatical, topic-consistent English for a few sentences, in the register of educational web text.
- ✅ Coarse associations: it knows HIV is a virus rather than a mineral, ice is frozen water, and a week has seven days.
- ❌ Specific facts and relations (capital cities, which body orbits which), arithmetic beyond memorised cases like "2 + 2 = 4", and following instructions.
- ❌ It invents names, numbers and "facts" fluently, and falls into repetition loops at low temperature. Use top-p ≈ 0.9 and a repetition penalty of about 1.1–1.2.

## Usage

**llama.cpp, Ollama or LM Studio** (GGUF, llama.cpp's built-in `deepseek2` architecture):

```sh
llama-completion -m {args.name}-q8_0.gguf -p "Photosynthesis is" -n 100 --temp 0.7 --top-p 0.9 --repeat-penalty 1.15
```

**PyTorch** (needs the `deepseek_moe` package from {f"[the training code]({args.code_url})" if args.code_url else "the training code"}):

```python
import json, torch
from safetensors.torch import load_file
from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.tokenizer import BPETokenizer

model = DeepSeekMoEModel(ModelConfig.from_dict(json.load(open("config.json")))).eval()
model.load_state_dict(load_file("model.safetensors"))
tok = BPETokenizer.from_file("tokenizer.json")
ids = torch.tensor([[tok.eot_id] + tok.encode("Photosynthesis is")])
print(tok.decode(model.generate(ids, 60, temperature=0.7, top_p=0.9, repetition_penalty=1.15)[0].tolist()))
```

## Files

| File | Contents |
|---|---|
{file_rows}

## Acknowledgements

- The architecture follows **DeepSeek-V3** (DeepSeek-AI, 2024): MLA, DeepSeekMoE, auxiliary-loss-free balancing and multi-token prediction.
- Pretraining data: **FineWeb-Edu** (Hugging Face, ODC-BY 1.0).

*Card generated {time.strftime('%Y-%m-%d')} by `prepare_release.py`.*
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/mini.pt")
    ap.add_argument("--size", default="mini", help="size label for the name, e.g. mini, small, base")
    ap.add_argument("--name", default=None, help="default: ShallowSeek-<size>-<base|chat>")
    ap.add_argument("--repo", default=None, help="target HF repo id, e.g. user/deepseek-mini-base (shown in the card)")
    ap.add_argument("--code-url", default=None, help="link to the training code (e.g. your GitHub repo)")
    ap.add_argument("--base-ckpt", default="checkpoints/mini.pt", help="IT cards: the base checkpoint to compare against")
    ap.add_argument("--base-repo", default=None, help="IT cards: HF repo of the base model (default: <repo> with -it -> -base)")
    ap.add_argument("--sft-log", default="checkpoints/mini-sft.log", help="IT cards: sft.py log for data and loss figures")
    ap.add_argument("--train-log", default="checkpoints/mini.log")
    ap.add_argument("--data-meta", default="data/fineweb-edu-small/meta.json")
    ap.add_argument("--out-dir", default=None, help="default: release/<name>")
    ap.add_argument("--gguf", nargs="*", default=["f16", "q8_0"], choices=["f32", "f16", "q8_0"])
    ap.add_argument("--gguf-repo", action="store_true",
                    help="also build release/<name>-GGUF: GGUF files only, so the Hub shows its GGUF panel "
                         "(repos that also hold safetensors show the safetensors panel instead)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    is_it = bool(ck.get("chat"))
    args.name = args.name or f"ShallowSeek-{args.size}-{'it' if is_it else 'base'}"
    out = Path(args.out_dir or f"release/{args.name}")
    out.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig.from_dict(ck["config"])
    step = ck.get("step")
    counts = DeepSeekMoEModel(cfg).param_counts()

    # Weights only (no optimizer state), in our own tensor naming.
    weights = {k: v.contiguous() for k, v in ck["model"].items()}
    save_file(weights, out / "model.safetensors", metadata={"format": "pt", "step": str(step)})
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out / "tokenizer.json").write_text(ck["tokenizer"]["serialized"])
    files = [("model.safetensors", "PyTorch weights (including the MTP module), our own parameter names"),
             ("config.json", "`ModelConfig` for `deepseek_moe`"),
             ("tokenizer.json", "Byte-level BPE tokenizer (Hugging Face `tokenizers` format)")]
    if ck.get("chat"):
        from deepseek_moe.chat import CHAT_TOKENS, END
        from deepseek_moe.tokenizer import EOT
        from export_gguf import CHAT_TEMPLATE
        (out / "chat_template.jinja").write_text(CHAT_TEMPLATE)
        (out / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": CHAT_TEMPLATE,
            "bos_token": None, "add_bos_token": False,  # SFT conversations start without <|endoftext|>
            "eos_token": END, "pad_token": EOT,
            "additional_special_tokens": [t for t in CHAT_TOKENS if t != END],
            "model_max_length": cfg.max_seq_len,
        }, indent=2))
        files += [("chat_template.jinja", "Chat template (Jinja), the same one embedded in the GGUF"),
                  ("tokenizer_config.json", "Special tokens (`<|end|>` = end of turn), chat template, context length")]
    for dtype in args.gguf:
        gguf_path = out / f"{args.name}-{dtype}.gguf"
        subprocess.run([sys.executable, "export_gguf.py", "--ckpt", args.ckpt, "--dtype", dtype,
                        "--name", args.name, "--out", str(gguf_path)], check=True)
        files.append((gguf_path.name, f"GGUF ({dtype}) for llama.cpp / Ollama / LM Studio"))

    tok_meta = json.loads(Path(args.data_meta).read_text()) if Path(args.data_meta).exists() else {}
    vals = val_curve(args.train_log)
    bench = latest("benchmarks.jsonl", step, args.ckpt)
    mmlu = latest("mmlu_results.jsonl", step, args.ckpt)
    if is_it:
        args.repo = args.repo or f"<user>/{args.name}"
        args.base_repo = args.base_repo or args.repo.replace("-it", "-base")
        base_step = torch.load(args.base_ckpt, map_location="cpu").get("step")
        base_bench = latest("benchmarks.jsonl", base_step, args.base_ckpt)
        base_mmlu = latest("mmlu_results.jsonl", base_step, args.base_ckpt)
        md = card_it(args, cfg, counts, sft_stats(args.sft_log), base_bench, base_mmlu, bench, mmlu, files)
    else:
        md = card(args, cfg, counts, step, tok_meta, vals, bench, mmlu, files)
    (out / "README.md").write_text(md)

    if args.gguf_repo:
        gg = out.parent / f"{args.name}-GGUF"
        gg.mkdir(parents=True, exist_ok=True)
        for name, _ in files:
            if name.endswith(".gguf"):
                (gg / name).write_bytes((out / name).read_bytes())
        main_md = (out / "README.md").read_text()
        base_id = args.repo or f"<user>/{args.name}"
        front, body = main_md.split("---\n", 2)[1], main_md.split("---\n", 2)[2]
        front = "".join(l + "\n" for l in front.splitlines()
                        if not l.startswith(("base_model:", "base_model_relation:", "library_name:")))
        front += (f"base_model: {base_id}\nbase_model_relation: quantized\nlibrary_name: gguf\n"
                  f"quantized_by: {base_id.split('/')[0]}\n")
        body = body.replace(f"# {args.name}\n", f"# {args.name}-GGUF\n\nGGUF builds of [{args.name}](https://huggingface.co/{base_id}) "
                            "for llama.cpp, Ollama and LM Studio. The PyTorch weights (`model.safetensors`), config and "
                            "tokenizer are in the main repo.\n\n", 1)
        (gg / "README.md").write_text(f"---\n{front}---\n{body}")
        if is_it:  # Ollama picks these up for `ollama run hf.co/<repo>`
            (gg / "template").write_text(OLLAMA_TEMPLATE)
            (gg / "chat_template.jinja").write_text((out / "chat_template.jinja").read_text())
            (gg / "params").write_text(json.dumps(OLLAMA_PARAMS, indent=2))
        print(f"GGUF repo folder: {gg}  ({', '.join(p.name for p in sorted(gg.iterdir()))})")

    print(f"\nrelease folder: {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:40s} {p.stat().st_size / 1e6:8.1f} MB")
    print(f"card metrics: val loss {'%.3f' % vals[-1] if vals else 'n/a'}, "
          f"benchmark {'yes' if bench else 'none for step ' + str(step)}, mmlu {'yes' if mmlu else 'none for step ' + str(step)}")
    repo = args.repo or "<user>/" + args.name
    print(f"\nto publish (after `hf auth login`):\n  hf upload {repo} {out} . --repo-type model")
    if args.gguf_repo:
        print(f"  hf upload {repo}-GGUF {out.parent / (args.name + '-GGUF')} . --repo-type model")


if __name__ == "__main__":
    main()
