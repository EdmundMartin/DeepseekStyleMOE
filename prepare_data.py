"""Build a Chinchilla-sized, pre-tokenized FineWeb-Edu slice for a preset.

Steps:
  1. Size the target: tokens = tokens_per_param (20, Chinchilla) x total model
     params, counted with the BPE vocab the model will actually use.
  2. Download FineWeb-Edu parquet shards (whole files; much faster than
     streaming) into the Hugging Face cache, only as many as needed.
  3. Train a byte-level BPE on the first --tokenizer-sample-mb of text.
  4. Tokenize documents (EOT-separated) into val.bin, then train.bin, as
     uint16, stopping once the target is reached.

Output: <out-dir>/{train.bin, val.bin, tokenizer.json, meta.json}, consumed by
    python train.py --data-dir <out-dir> --preset <preset>

    python prepare_data.py --preset small
    python prepare_data.py --preset tiny --out-dir data/fineweb-edu-tiny
    python prepare_data.py --preset small --target-tokens 50_000_000   # explicit size
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.tokenizer import DEFAULT_BPE_VOCAB, BPETokenizer


def iter_documents(repo: str, subset: str) -> Iterator[str]:
    shards = sorted(
        f.path for f in HfApi().list_repo_tree(repo, path_in_repo=subset, repo_type="dataset")
        if f.path.endswith(".parquet")
    )
    for shard in shards:
        print(f"downloading {shard} (cached under ~/.cache/huggingface)")
        local = hf_hub_download(repo, shard, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(columns=["text"], batch_size=1024):
            yield from batch.column("text").to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small")
    ap.add_argument("--vocab-size", type=int, default=None, help="default: per preset")
    ap.add_argument("--tokens-per-param", type=float, default=20.0, help="Chinchilla ratio")
    ap.add_argument("--target-tokens", type=int, default=None, help="override the Chinchilla target")
    ap.add_argument("--val-tokens", type=int, default=2_000_000)
    ap.add_argument("--tokenizer-sample-mb", type=float, default=200.0)
    ap.add_argument("--repo", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample/10BT")
    ap.add_argument("--out-dir", default=None, help="default: data/fineweb-edu-<preset>")
    args = ap.parse_args()

    vocab = args.vocab_size or DEFAULT_BPE_VOCAB.get(args.preset, 8192)
    counts = DeepSeekMoEModel(ModelConfig.from_preset(args.preset, vocab_size=vocab)).param_counts()
    target = args.target_tokens or int(args.tokens_per_param * counts["total"])
    out = Path(args.out_dir or f"data/fineweb-edu-{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.preset}: {counts['total'] / 1e6:.2f}M params (vocab {vocab}) -> "
          f"target {target / 1e6:.0f}M train tokens + {args.val_tokens / 1e6:.0f}M val")

    docs = iter_documents(args.repo, args.subset)

    # 1) Tokenizer on a sample; keep the sampled docs so they're tokenized too.
    sample, sample_bytes = [], 0
    while sample_bytes < args.tokenizer_sample_mb * 1e6:
        text = next(docs)
        sample.append(text)
        sample_bytes += len(text.encode("utf-8"))
    print(f"training BPE on {len(sample):,} docs ({sample_bytes / 1e6:.0f}MB)")
    tok = BPETokenizer.train_from_texts(sample, vocab)
    tok.save(out / "tokenizer.json")
    assert tok.vocab_size <= 65536, "uint16 storage needs vocab <= 65536"

    # 2) Tokenize: first val_tokens -> val.bin, then target tokens -> train.bin.
    def all_docs() -> Iterator[str]:
        yield from sample
        yield from docs

    eot = tok.eot_id
    written = {"val": 0, "train": 0}
    n_docs, n_bytes, t0 = 0, 0, time.time()
    files = {k: open(out / f"{k}.bin", "wb") for k in written}
    batch: list[str] = []
    try:
        for text in all_docs():
            batch.append(text)
            if len(batch) < 512:
                continue
            for ids in tok.encode_batch(batch):
                ids.append(eot)
                split = "val" if written["val"] < args.val_tokens else "train"
                if split == "train":
                    ids = ids[: target - written["train"]]
                files[split].write(np.asarray(ids, dtype=np.uint16).tobytes())
                written[split] += len(ids)
            n_docs += len(batch)
            n_bytes += sum(len(t.encode("utf-8")) for t in batch)
            batch = []
            if n_docs % (512 * 50) == 0:
                done = written["train"] / target
                print(f"  {n_docs:,} docs  train {written['train'] / 1e6:.1f}M/{target / 1e6:.0f}M "
                      f"({done:.0%})  {time.time() - t0:.0f}s")
            if written["train"] >= target:
                break
    finally:
        for f in files.values():
            f.close()

    total = written["train"] + written["val"]
    meta = {
        "preset": args.preset, "vocab_size": tok.vocab_size, "params_total": counts["total"],
        "tokens_per_param": written["train"] / counts["total"], "train_tokens": written["train"],
        "val_tokens": written["val"], "documents": n_docs, "bytes_per_token": n_bytes / total,
        "source": f"{args.repo}/{args.subset}", "dtype": "uint16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if written["train"] < target:
        print(f"WARNING: ran out of data at {written['train']:,} of {target:,} tokens")


if __name__ == "__main__":
    main()
