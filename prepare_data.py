"""Build a pre-tokenized pretraining set for a preset, optionally mixing several sources.

Steps:
  1. Size the target: tokens = tokens_per_param (20, Chinchilla) x total model params,
     counted with the BPE vocab the model will actually use (or --target-tokens).
  2. Stream documents from Hugging Face parquet shards (whole-file downloads into the
     HF cache; much faster than `datasets` streaming), only as many as needed.
  3. Train a byte-level BPE on a sample drawn from the mix in proportion to its weights.
  4. Tokenize documents (EOT-separated) into val.bin, then train.bin, as uint16,
     interleaving sources so each keeps its share of tokens throughout.

Output: <out-dir>/{train.bin, val.bin, tokenizer.json, meta.json}, consumed by
    python train.py --data-dir <out-dir> --preset <preset>

    python prepare_data.py --preset small                                   # FineWeb-Edu only (default)
    python prepare_data.py --preset base --mix fineweb-edu=0.75 code=0.15 smoltalk=0.10
    python prepare_data.py --preset base --mix fineweb-edu=0.7 cosmopedia=0.1 code=0.1 smoltalk=0.1
    python prepare_data.py --list-sources

Conversations (smoltalk) are rendered in the chat format of deepseek_moe/chat.py and the
chat tokens are added to the tokenizer, so later SFT needs no vocab resize.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from deepseek_moe import ModelConfig
from deepseek_moe.chat import CHAT_TOKENS, END, ROLE_TOKENS
from deepseek_moe.tokenizer import DEFAULT_BPE_VOCAB, BPETokenizer

# Permissive licences only, for the code source.
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "unlicense", "cc0-1.0", "isc"}
CODE_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "GO", "Rust", "Shell", "SQL"}
MAX_CODE_BYTES = 50_000  # skip minified / generated monsters


def _render_chat(messages: list[dict]) -> str:
    return "".join(f"{ROLE_TOKENS[m['role']]}{m['content']}{END}" for m in messages if m["role"] in ROLE_TOKENS)


def _code_rows(batch) -> Iterator[str]:
    cols = batch.to_pydict()
    for code, lang, lic, size in zip(cols["code"], cols["language"], cols["license"], cols["size"]):
        if lang in CODE_LANGUAGES and lic in PERMISSIVE and size <= MAX_CODE_BYTES:
            yield code


# name -> (repo, shard path prefix, columns to read, row -> text documents)
SOURCES: dict[str, tuple[str, str, list[str], Callable]] = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", ["text"],
                    lambda b: iter(b.column("text").to_pylist())),
    "fineweb-edu-100bt": ("HuggingFaceFW/fineweb-edu", "sample/100BT/", ["text"],
                          lambda b: iter(b.column("text").to_pylist())),
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "data/", ["text"],
                   lambda b: iter(b.column("text").to_pylist())),
    "code": ("codeparrot/github-code-clean", "data/", ["code", "language", "license", "size"], _code_rows),
    "smoltalk": ("HuggingFaceTB/smoltalk", "data/all/train-", ["messages"],
                 lambda b: (_render_chat(m) for m in b.column("messages").to_pylist())),
}


def iter_source(name: str, skip: int = 0) -> Iterator[str]:
    """Yield documents from a source's parquet shards, skipping the first `skip` (already used) ones."""
    repo, prefix, columns, to_docs = SOURCES[name]
    shards = sorted(
        s.rfilename for s in HfApi().dataset_info(repo).siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"no parquet shards for {name} ({repo}/{prefix})")
    for shard in shards:
        print(f"  [{name}] downloading {shard}", flush=True)
        local = hf_hub_download(repo, shard, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(columns=columns, batch_size=1024):
            for text in to_docs(batch):
                if text:
                    if skip > 0:
                        skip -= 1
                        continue
                    yield text


def parse_mix(items: list[str]) -> dict[str, float]:
    mix = {}
    for item in items:
        name, _, w = item.partition("=")
        if name not in SOURCES:
            raise SystemExit(f"unknown source {name!r}; choose from {sorted(SOURCES)}")
        mix[name] = float(w or 1)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--mix", nargs="+", default=["fineweb-edu=1"], metavar="SOURCE=WEIGHT",
                    help="token shares per source (normalised)")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=None, help="default: per preset")
    ap.add_argument("--tokens-per-param", type=float, default=20.0, help="Chinchilla ratio")
    ap.add_argument("--target-tokens", type=float, default=None, help="override the Chinchilla target")
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=200.0)
    ap.add_argument("--out-dir", default=None, help="default: data/fineweb-edu-<preset> (or data/mix-<preset>)")
    ap.add_argument("--tokenizer-path", default=None, help="reuse this tokenizer.json instead of training one")
    ap.add_argument("--skip-docs", nargs="*", default=[], metavar="SOURCE=N",
                    help="skip the first N documents of a source (e.g. ones an earlier dataset already used)")
    ap.add_argument("--extend", default=None, metavar="DIR",
                    help="grow an existing dataset: reuse its tokenizer and val.bin, skip the documents it used "
                         "(single-source fineweb-edu), and write its train.bin + the new tokens to --out-dir. "
                         "--target-tokens is then the combined total")
    args = ap.parse_args()

    if args.list_sources:
        for name, (repo, prefix, _, _) in SOURCES.items():
            print(f"{name:18s} {repo}/{prefix}")
        return

    mix = parse_mix(args.mix)
    skips = {k: int(v) for k, _, v in (x.partition("=") for x in args.skip_docs)}
    base_meta, base_train = {}, 0
    if args.extend:
        ext = Path(args.extend)
        base_meta = json.loads((ext / "meta.json").read_text())
        base_train = base_meta["train_tokens"]
        args.tokenizer_path = args.tokenizer_path or str(ext / "tokenizer.json")
        args.val_tokens = 0  # keep the original validation set so losses stay comparable
        if list(mix) == ["fineweb-edu"] and "fineweb-edu" not in skips:
            skips["fineweb-edu"] = base_meta["documents"]
    chat_in_mix = "smoltalk" in mix
    vocab = args.vocab_size or DEFAULT_BPE_VOCAB.get(args.preset, 8192)
    total_params = ModelConfig.from_preset(args.preset, vocab_size=vocab).param_estimate()["total"]
    target = int(args.target_tokens or args.tokens_per_param * total_params) - base_train  # new tokens to write
    if target <= 0:
        raise SystemExit(f"--target-tokens must exceed the {base_train:,} tokens already in {args.extend}")
    default_dir = f"data/fineweb-edu-{args.preset}" if list(mix) == ["fineweb-edu"] else f"data/mix-{args.preset}"
    out = Path(args.out_dir or default_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.preset}: {total_params / 1e6:.2f}M params (vocab {vocab}) -> target {target / 1e6:,.0f}M train "
          f"tokens + {args.val_tokens / 1e6:.0f}M val | mix " + ", ".join(f"{k} {v:.0%}" for k, v in mix.items()))

    if skips:
        print("skipping already-used documents: " + ", ".join(f"{k} {v:,}" for k, v in skips.items()))
    streams = {name: iter_source(name, skips.get(name, 0)) for name in mix}
    buffered: dict[str, list[str]] = {name: [] for name in mix}

    if args.tokenizer_path:
        tok = BPETokenizer.from_file(args.tokenizer_path)
        print(f"reusing tokenizer {args.tokenizer_path} (vocab {tok.vocab_size})")
        tok.save(out / "tokenizer.json")
        mix_sample = False
    else:
        mix_sample = True

    # 1) Tokenizer on a sample drawn from each source in proportion to its weight.
    sample_bytes = {name: 0 for name in mix}
    for name, w in (mix.items() if mix_sample else []):
        while sample_bytes[name] < args.tokenizer_sample_mb * 1e6 * w:
            text = next(streams[name])
            buffered[name].append(text)
            sample_bytes[name] += len(text.encode("utf-8"))
    if mix_sample:
        sample = [t for docs in buffered.values() for t in docs]
        print(f"training BPE (vocab {vocab}) on {len(sample):,} docs ({sum(sample_bytes.values()) / 1e6:.0f}MB)"
              f"{' with chat tokens' if chat_in_mix else ''}", flush=True)
        tok = BPETokenizer.train_from_texts(sample, vocab, extra_special=CHAT_TOKENS if chat_in_mix else ())
        tok.save(out / "tokenizer.json")
        del sample
    assert tok.vocab_size <= 65536, "uint16 storage needs vocab <= 65536"

    def next_batch(name: str, n: int = 64) -> list[str]:
        batch = buffered[name][:n]
        buffered[name] = buffered[name][n:]
        for text in streams[name]:
            if len(batch) >= n:
                break
            batch.append(text)
        return batch

    # 2) Interleave: always draw from the source furthest below its share of tokens written so far.
    eot = tok.eot_id
    written = {"val": 0, "train": 0}
    per_source = {name: 0 for name in mix}
    exhausted: set[str] = set()
    n_docs, n_bytes, t0, last_report = 0, 0, time.time(), 0
    if args.extend:
        import shutil
        shutil.copyfile(Path(args.extend) / "train.bin", out / "train.bin")  # new tokens are appended after these
        shutil.copyfile(Path(args.extend) / "val.bin", out / "val.bin")
        files = {"train": open(out / "train.bin", "ab"), "val": open(out / "val.bin", "ab")}
    else:
        files = {k: open(out / f"{k}.bin", "wb") for k in written}
    try:
        while written["train"] < target and len(exhausted) < len(mix):
            total = sum(per_source.values()) + 1
            name = max((n for n in mix if n not in exhausted), key=lambda n: mix[n] * total - per_source[n])
            batch = next_batch(name)
            if not batch:
                exhausted.add(name)
                print(f"  WARNING: {name} ran out of data", flush=True)
                continue
            for ids in tok.encode_batch(batch):
                ids.append(eot)
                split = "val" if written["val"] < args.val_tokens else "train"
                if split == "train":
                    ids = ids[: target - written["train"]]
                files[split].write(np.asarray(ids, dtype=np.uint16).tobytes())
                written[split] += len(ids)
                per_source[name] += len(ids)
                if written["train"] >= target:
                    break
            n_docs += len(batch)
            n_bytes += sum(len(t.encode("utf-8")) for t in batch)
            if written["train"] - last_report >= max(target // 50, 1):
                last_report = written["train"]
                shares = ", ".join(f"{k} {v / max(sum(per_source.values()), 1):.1%}" for k, v in per_source.items())
                print(f"  {n_docs:,} docs  train {written['train'] / 1e6:,.1f}M/{target / 1e6:,.0f}M "
                      f"({written['train'] / target:.0%})  [{shares}]  {time.time() - t0:.0f}s", flush=True)
    finally:
        for f in files.values():
            f.close()

    total = written["train"] + written["val"]
    meta = {
        "preset": args.preset, "vocab_size": tok.vocab_size, "params_total": total_params,
        "tokens_per_param": (base_train + written["train"]) / total_params,
        "train_tokens": base_train + written["train"],
        "val_tokens": base_meta.get("val_tokens", 0) + written["val"],
        "documents": base_meta.get("documents", 0) + n_docs, "bytes_per_token": n_bytes / max(total, 1),
        **({"extended_from": {"dir": args.extend, "train_tokens": base_train, "documents": base_meta["documents"]},
            "new_train_tokens": written["train"], "new_documents": n_docs, "skipped_documents": skips}
           if args.extend else {}),
        "mix_weights": mix, "mix_tokens": per_source, "chat_tokens_in_vocab": chat_in_mix,
        "sources": {k: f"{SOURCES[k][0]}/{SOURCES[k][1]}" for k in mix}, "dtype": "uint16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if written["train"] < target:
        print(f"WARNING: ran out of data at {written['train']:,} of {target:,} tokens")


if __name__ == "__main__":
    main()
