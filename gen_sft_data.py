"""Generate grounded SFT question/answer pairs from the pretraining corpus.

Documents are read back out of the pre-tokenized train.bin (split on
<|endoftext|>), so every question is about text the model actually saw during
pretraining. A larger "teacher" model behind any OpenAI-compatible endpoint
writes the pairs: vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter,
OpenAI, etc.

    # local Ollama
    python gen_sft_data.py --base-url http://localhost:11434/v1 --model qwen2.5:14b --docs 2000

    # Qwen3-style reasoning models: turn thinking off, or it eats the token budget
    python gen_sft_data.py --base-url https://<host>/v1 --api-key-file token --model Qwen3.8-9B-mlx-4Bit --no-think

    # hosted endpoint; key read from $OPENAI_API_KEY (or --api-key-env NAME, or --api-key-file PATH)
    python gen_sft_data.py --base-url https://openrouter.ai/api/v1 --model <model-id> \\
        --api-key-env OPENROUTER_API_KEY --docs 5000 --concurrency 16

Output is JSONL, one conversation per line: {"messages": [...], "doc_index": i}.
Re-running with the same --out skips documents already processed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

from deepseek_moe.tokenizer import BPETokenizer

SYSTEM_PROMPT = "You create high-quality training data for a small language model. You reply with JSON only."

USER_PROMPT = """Below is a document. Write {n} question-answer pairs grounded in it.

Rules:
- Each question must make sense on its own, to someone who has never seen the document. Never refer to "the document", "the text", "the passage", or "the author".
- Each answer must be fully supported by the document and must also stand alone: no references to the document.
- Vary the questions: facts, explanations ("why"/"how"), definitions, comparisons.
- Answers: clear, plain prose, 1-4 sentences, at most {max_words} words.
- If the document has no substantive content worth asking about (e.g. navigation text, ads, lists of links), return an empty list.

Reply with only this JSON, no other text:
{{"pairs": [{{"question": "...", "answer": "..."}}]}}

Document:
\"\"\"
{document}
\"\"\""""

_BANNED = re.compile(r"\b(the|this) (document|text|passage|article)\b|\bthe author\b", re.IGNORECASE)


def split_documents(data: np.ndarray, eot_id: int) -> list[tuple[int, int]]:
    """(start, end) token spans of each EOT-separated document in a flat token array."""
    bounds = np.flatnonzero(data == eot_id)
    starts = np.concatenate([[0], bounds + 1])
    ends = np.concatenate([bounds, [len(data)]])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def parse_pairs(content: str) -> list[dict]:
    """Extract Q&A pairs from a model reply, tolerating code fences and reasoning tags."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    pairs = obj.get("pairs", []) if isinstance(obj, dict) else []
    out = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        q, a = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
        # Drop pairs that leak the document framing; the student never sees the document.
        if q and a and not _BANNED.search(q) and not _BANNED.search(a):
            out.append({"question": q, "answer": a})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key (local servers: any)")
    ap.add_argument("--api-key-file", default=None, help="file containing just the key (overrides --api-key-env)")
    ap.add_argument("--data-dir", default="data/fineweb-edu-small")
    ap.add_argument("--out", default=None, help="default: <data-dir>/sft_qa.jsonl")
    ap.add_argument("--docs", type=int, default=1000, help="number of documents to process")
    ap.add_argument("--pairs-per-doc", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=100, help="answer length cap given to the teacher")
    ap.add_argument("--min-doc-tokens", type=int, default=150)
    ap.add_argument("--max-doc-chars", type=int, default=6000, help="documents are truncated to this for the prompt")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=8, help="per-document retries with backoff (~5 min)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048, help="teacher reply cap (raise for reasoning models)")
    ap.add_argument("--json-mode", action="store_true", help="send response_format=json_object (if supported)")
    ap.add_argument("--no-think", action="store_true",
                    help="disable reasoning via chat_template_kwargs.enable_thinking=false (Qwen3-style models)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.data_dir)
    out_path = Path(args.out or d / "sft_qa.jsonl")
    tok = BPETokenizer.from_file(d / "tokenizer.json")
    data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    spans = [s for s in split_documents(data, tok.eot_id) if s[1] - s[0] >= args.min_doc_tokens]
    order = list(range(len(spans)))
    random.Random(args.seed).shuffle(order)

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.open():
            done.add(json.loads(line)["doc_index"])
    todo = [i for i in order if i not in done][: max(0, args.docs - len(done))]
    print(f"{len(spans):,} eligible documents; {len(done):,} already processed; {len(todo):,} to do")
    if not todo:
        return

    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text().strip()
    else:
        api_key = os.environ.get(args.api_key_env) or "not-needed"
    client = OpenAI(base_url=args.base_url, api_key=api_key)
    extra = {"response_format": {"type": "json_object"}} if args.json_mode else {}
    if args.no_think:
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    lock = threading.Lock()
    stats = {"docs": 0, "pairs": 0, "empty": 0, "errors": 0}

    def work(doc_i: int) -> tuple[int, list[dict]]:
        s, e = spans[doc_i]
        text = tok.decode(data[s:e].tolist())[: args.max_doc_chars]
        # Ride out endpoint outages (e.g. a tunnel dropping) with exponential backoff, ~5 minutes total.
        for attempt in range(args.retries + 1):
            try:
                return doc_i, ask(text)
            except Exception:
                if attempt == args.retries:
                    raise
                time.sleep(min(60, 2 ** attempt * 2))

    def ask(text: str) -> list[dict]:
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(
                    n=args.pairs_per_doc, max_words=args.max_words, document=text)},
            ],
            **extra,
        )
        return parse_pairs(resp.choices[0].message.content or "")

    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(work, i) for i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                doc_i, pairs = fut.result()
            except Exception as exc:  # keep going; failed docs are retried on the next run
                stats["errors"] += 1
                print(f"  error: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            with lock:
                stats["docs"] += 1
                stats["pairs"] += len(pairs)
                stats["empty"] += not pairs
                # One line per pair, plus a marker line so the doc is skipped on resume if it yielded nothing.
                for p in pairs:
                    fh.write(json.dumps({"messages": [
                        {"role": "user", "content": p["question"]},
                        {"role": "assistant", "content": p["answer"]},
                    ], "doc_index": doc_i}) + "\n")
                if not pairs:
                    fh.write(json.dumps({"messages": [], "doc_index": doc_i}) + "\n")
                fh.flush()
            if n % 25 == 0 or n == len(futures):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(futures)} docs  {stats['pairs']} pairs  {stats['empty']} empty  "
                      f"{stats['errors']} errors  {rate:.1f} docs/s", flush=True)
    print(f"wrote {out_path}: {stats}")


if __name__ == "__main__":
    main()
