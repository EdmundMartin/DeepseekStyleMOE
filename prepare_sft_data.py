"""Filter HuggingFaceTB/smol-smoltalk into short general-chat SFT data for a small model.

Keeps general conversation / instruction-following sources, drops code, summarisation and
long-document tasks, and keeps only conversations that fit *whole* in the model's context
window (measured with its own tokenizer and chat template). Writes JSONL in the same
{"messages": [...]} format as gen_sft_data.py, ready for sft.py --data.

    python prepare_sft_data.py --tokenizer data/fineweb-edu-small/tokenizer.json --max-len 1024 --limit 6000
    (all everyday-conversations small talk is kept on top of the 6,000; --include-all to change)
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from deepseek_moe.chat import ROLE_TOKENS
from deepseek_moe.tokenizer import BPETokenizer

REPO = "HuggingFaceTB/smol-smoltalk"
# General chat / instruction following. Excluded: self-oss-instruct (code), longalign (long docs),
# smol-summarize-* and smollm-rewrite-30k (need long input passages).
KEEP_SOURCES = {"smol-magpie-ultra-short", "everyday-conversations", "openhermes-50k",
                "smol-contraints", "explore-instruct-rewrite"}


def token_counts(tok: BPETokenizer, messages: list[dict]) -> tuple[int, int]:
    """(total tokens, assistant tokens trained on) under deepseek_moe/chat.py's template:
    role marker + content + <|end|> per turn; the loss covers assistant content + <|end|>."""
    total = assistant = 0
    for m in messages:
        n = len(tok.encode(m["content"]))
        total += n + 2
        if m["role"] == "assistant":
            assistant += n + 1
    return total, assistant


def parse_caps(items: list[str]) -> dict[str, float]:
    caps = {}
    for item in items:
        name, _, frac = item.partition("=")
        caps[name] = float(frac)
    return caps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="data/fineweb-edu-small/tokenizer.json")
    ap.add_argument("--max-len", type=int, default=1024, help="model context window")
    ap.add_argument("--limit", type=int, default=6000, help="conversations to sample from the other sources")
    ap.add_argument("--include-all", nargs="*", default=["everyday-conversations"],
                    help="sources kept in full, on top of --limit (default: everyday small talk)")
    ap.add_argument("--token-share-cap", nargs="*", default=["smol-magpie-ultra-short=0.15"], metavar="SOURCE=FRAC",
                    help="max share of assistant tokens (across this file + --also-count) for a source")
    ap.add_argument("--also-count", nargs="*", default=["data/fineweb-edu-small/sft_qa.jsonl"],
                    help="other SFT JSONL files in the final mix, counted when applying --token-share-cap")
    ap.add_argument("--out", default="data/sft/smol_smoltalk_filtered.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = BPETokenizer.from_file(args.tokenizer)
    shards = sorted(s.rfilename for s in HfApi().dataset_info(REPO).siblings
                    if s.rfilename.startswith("data/train-") and s.rfilename.endswith(".parquet"))
    seen, kept = collections.Counter(), collections.defaultdict(list)
    drop = collections.Counter()
    for shard in shards:
        print(f"downloading {shard}", flush=True)
        for row in pq.read_table(hf_hub_download(REPO, shard, repo_type="dataset")).to_pylist():
            src, msgs = row["source"], row["messages"]
            seen[src] += 1
            if src not in KEEP_SOURCES:
                drop["source"] += 1
                continue
            msgs = [m for m in msgs if m["role"] in ROLE_TOKENS]
            if not msgs or msgs[-1]["role"] != "assistant":
                drop["no assistant reply"] += 1
                continue
            if any("```" in m["content"] for m in msgs):
                drop["code block"] += 1
                continue
            kept[src].append(msgs)

    # Sample first, then tokenise lazily: only the conversations we might keep get tokenised.
    rng = random.Random(args.seed)

    def take(convs: list[list[dict]], src: str, limit: int | None) -> list[dict]:
        out = []
        for msgs in convs:
            if limit is not None and len(out) >= limit:
                break
            total, asst = token_counts(tok, msgs)
            if total <= args.max_len:
                out.append({"messages": msgs, "source": src, "_asst": asst})
            else:
                drop["too long"] += 1
        return out

    fit = []
    for src in args.include_all:
        fit += take(kept.get(src, []), src, None)
    pool = [(src, m) for src, convs in kept.items() if src not in args.include_all for m in convs]
    rng.shuffle(pool)
    rest: list[dict] = []
    for src, msgs in pool:
        if len(rest) >= args.limit:
            break
        rest += take([msgs], src, None)
    fit += rest
    for src in kept:
        print(f"  {src:26s} {seen[src]:7,} rows -> {len(kept[src]):7,} after filters -> "
              f"{sum(r['source'] == src for r in fit):6,} sampled")

    # Cap chosen sources' share of assistant tokens across the whole SFT mix.
    extra = 0
    for path in args.also_count:
        for line in Path(path).read_text().splitlines():
            msgs = json.loads(line)["messages"]
            if msgs:
                extra += token_counts(tok, msgs)[1]
    for src, cap in parse_caps(args.token_share_cap).items():
        others = extra + sum(r["_asst"] for r in fit if r["source"] != src)
        allowed = cap * others / (1 - cap)
        capped, used = [], 0
        for r in (r for r in fit if r["source"] == src):
            if used + r["_asst"] > allowed:
                continue
            capped.append(r)
            used += r["_asst"]
        before = sum(r["source"] == src for r in fit)
        fit = [r for r in fit if r["source"] != src] + capped
        print(f"  capped {src} at {cap:.0%} of assistant tokens: {before:,} -> {len(capped):,} conversations "
              f"({used / 1e6:.2f}M tokens)")
    rng.shuffle(fit)

    total_asst = extra + sum(r["_asst"] for r in fit)
    share = collections.Counter()
    for r in fit:
        share[r["source"]] += r["_asst"]
    print(f"assistant-token shares of the full SFT mix ({total_asst / 1e6:.2f}M tokens):")
    if extra:
        print(f"  {'(--also-count files)':26s} {extra / total_asst:6.1%}")
    for src, n in share.most_common():
        print(f"  {src:26s} {n / total_asst:6.1%}")
    for r in fit:
        r.pop("_asst")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in fit:
            fh.write(json.dumps(row) + "\n")
    by_src = collections.Counter(r["source"] for r in fit)
    turns = collections.Counter(sum(m["role"] == "assistant" for m in r["messages"]) for r in fit)
    print(f"dropped: {dict(drop)}")
    print(f"wrote {len(fit):,} conversations to {out}")
    print(f"  by source: {dict(by_src.most_common())}")
    print(f"  assistant turns per conversation: {dict(sorted(turns.items()))}")


if __name__ == "__main__":
    main()
