"""Package the gen_sft_data.py Q&A as a Hugging Face dataset, with each pair's source passage.

Drops the empty marker rows (documents the teacher found nothing to ask about) and attaches the
FineWeb-Edu passage each pair was generated from, decoded from train.bin via its doc_index (the
same text, truncated the same way, that the teacher model saw).

    python export_qa_dataset.py --repo edededdy/fineweb-edu-grounded-qa
    hf upload edededdy/fineweb-edu-grounded-qa release/datasets/fineweb-edu-grounded-qa . --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from deepseek_moe.tokenizer import BPETokenizer
from gen_sft_data import split_documents

MENTIONS_SOURCE = re.compile(r"\b(the|this) (document|text|passage|article)\b|\bthe author\b|according to the", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", default="data/fineweb-edu-small/sft_qa.jsonl")
    ap.add_argument("--data-dir", default="data/fineweb-edu-small")
    ap.add_argument("--repo", default="edededdy/fineweb-edu-grounded-qa")
    ap.add_argument("--teacher", default="Qwen3.8-9B (MLX 4-bit), thinking disabled")
    ap.add_argument("--min-doc-tokens", type=int, default=150, help="must match gen_sft_data.py")
    ap.add_argument("--max-doc-chars", type=int, default=6000, help="must match gen_sft_data.py")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    name = args.repo.split("/")[-1]
    out = Path(args.out_dir or f"release/datasets/{name}")
    (out / "data").mkdir(parents=True, exist_ok=True)

    d = Path(args.data_dir)
    tok = BPETokenizer.from_file(d / "tokenizer.json")
    data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    spans = [s for s in split_documents(data, tok.eot_id) if s[1] - s[0] >= args.min_doc_tokens]

    rows, empty, docs = [], 0, set()
    passage_cache: dict[int, str] = {}
    for line in open(args.qa):
        r = json.loads(line)
        if not r["messages"]:
            empty += 1
            continue
        i = r["doc_index"]
        if i not in passage_cache:
            s, e = spans[i]
            passage_cache[i] = tok.decode(data[s:e].tolist())[: args.max_doc_chars]
        q, a = r["messages"][0]["content"], r["messages"][1]["content"]
        rows.append({"question": q, "answer": a, "messages": r["messages"], "passage": passage_cache[i],
                     "doc_index": i, "refers_to_passage": bool(MENTIONS_SOURCE.search(q) or MENTIONS_SOURCE.search(a))})
        docs.add(i)

    with open(out / "data" / "train.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n, n_ref = len(rows), sum(r["refers_to_passage"] for r in rows)
    ans_words = sorted(len(r["answer"].split()) for r in rows)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    (out / "README.md").write_text(f"""---
license: odc-by
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- synthetic
- sft
- fineweb-edu
- reading-comprehension
- rag
size_categories:
- 10K<n<100K
source_datasets:
- HuggingFaceFW/fineweb-edu
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# FineWeb-Edu Grounded Q&A

**{n:,} question–answer pairs, each grounded in a FineWeb-Edu passage that's included alongside it.** They were generated to fine-tune [ShallowSeek-mini](https://huggingface.co/edededdy/ShallowSeek-mini-base), a tiny DeepSeek-style MoE language model, so that its SFT data would ask about text the model had actually seen during pretraining.

## How it was made

1. **Passages:** {len(docs):,} documents sampled at random from the FineWeb-Edu `sample/10BT` slice that ShallowSeek-mini was pretrained on ({meta.get('documents', 236544):,} documents, {meta.get('train_tokens', 286_400_000) / 1e6:.0f}M tokens). Only documents of at least {args.min_doc_tokens} tokens were eligible, and each was truncated to its first {args.max_doc_chars:,} characters. That truncated text is exactly what's in `passage`.
2. **Generation:** a teacher model, **{args.teacher}**, was asked for 3 question–answer pairs per passage, under these rules:
   - each question must make sense on its own, without seeing the passage;
   - each answer must be fully supported by the passage and stand alone, in 1–4 sentences of at most ~100 words;
   - questions should vary (facts, explanations, definitions, comparisons);
   - if the passage has nothing substantive (navigation text, ads, link lists), return no pairs.
3. **Filtering:** replies were parsed as JSON. Documents that produced no pairs ({empty} of them) appear only as empty marker rows in the raw generation log and **are excluded here**. No other filtering was applied.

## Fields

| Field | Description |
|---|---|
| `question` | The generated question |
| `answer` | The generated answer (median {ans_words[len(ans_words) // 2]} words) |
| `messages` | The same pair in chat format: `[{{"role": "user", ...}}, {{"role": "assistant", ...}}]` |
| `passage` | The FineWeb-Edu source text the pair was generated from (truncated as above) |
| `doc_index` | Index of the source document in the generation run; pairs from the same passage share it |
| `refers_to_passage` | `true` for the {n_ref:,} pairs ({n_ref / n:.1%}) whose question or answer refers to "the document", "the text" and so on, despite instructions. They're fine for passage-grounded use, but you may want to drop them for closed-book chat SFT |

## Uses

- **Chat SFT** (`messages`): short, factual, single-turn answers. Filter on `refers_to_passage == false` for closed-book use.
- **Reading comprehension / RAG** (`passage` + `question` → `answer`): every answer is checkable against its passage.
- **Studying grounding:** comparing a model's closed-book answers against the passage-supported reference.

## Caveats

- **Synthetic:** written by a 9B-parameter model and **not human-verified**. Most pairs are accurate to their passage, but expect some errors, oversimplifications and awkward phrasing.
- **Inherited from the web:** FineWeb-Edu is filtered educational web text, but passages can contain dated, regional or incorrect information, which the answers inherit.
- **Time-relative wording:** some questions concern a passage's own moment ("the recent eruption…"), which can be ambiguous out of context.
- **English only.**

## Licence and attribution

The passages come from **FineWeb-Edu** (Hugging Face), licensed under **ODC-BY 1.0**, and this dataset is released under the same licence. The question–answer pairs were generated by {args.teacher.split(' (')[0]}; check that model's licence terms if they matter for your use. Generation code: [gen_sft_data.py](https://github.com/EdmundMartin/DeepseekStyleMOE/blob/main/gen_sft_data.py).
""")
    size = (out / "data" / "train.jsonl").stat().st_size / 1e6
    print(f"{n:,} pairs from {len(docs):,} passages (dropped {empty} empty marker rows; "
          f"{n_ref} pairs flagged refers_to_passage) -> {out} ({size:.1f} MB)")
    print(f"upload:  hf upload {args.repo} {out} . --repo-type dataset")


if __name__ == "__main__":
    main()
