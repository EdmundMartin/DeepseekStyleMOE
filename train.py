"""Pre-train a DeepSeek-style MoE model on your corpus.

By default a byte-level BPE tokenizer is trained on the same text files first
(Hugging Face `tokenizers`), sized per preset unless --vocab-size is given.

    python train.py --preset tiny --steps 2000
    python train.py --data corpus/*.txt --preset small --vocab-size 16000
    python train.py --preset small --set n_mtp_modules=2 use_fp8=true --device cuda
    python train.py --tokenizer-path checkpoints/tokenizer.json   # reuse a trained BPE
    python train.py --tokenizer bytes                             # no BPE, one token per byte

Pre-tokenized data from prepare_data.py (one pass over the tokens by default,
checkpointing every --save-every steps; rerun with --resume to continue):

    python train.py --data-dir data/fineweb-edu-small --preset small --out checkpoints/small.pt
    python train.py --data-dir data/fineweb-edu-small --preset small --out checkpoints/small.pt --resume

GPU (e.g. RunPod): device and bf16 are picked automatically; add --compile, and use --grad-accum to
reach large token batches (tokens/step = batch-size x seq-len x grad-accum):

    python train.py --data-dir data/fineweb-edu-base --preset base --batch-size 16 --seq-len 2048 \
        --grad-accum 16 --compile --out checkpoints/base.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.runtime import autocast, pick_device, resolve_precision, setup_backends
from deepseek_moe.tokenizer import DEFAULT_BPE_VOCAB, BPETokenizer, ByteTokenizer, encode_files

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                out[key] = float(raw)
    return out


def resolve_data(paths: list[str] | None) -> list[str]:
    if paths:
        return paths
    p = Path("data/tinyshakespeare.txt")
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        print(f"downloading tiny shakespeare -> {p}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, p)
    return [str(p)]


def build_tokenizer(args, files: list[str]):
    if args.tokenizer == "bytes":
        return ByteTokenizer()
    if args.tokenizer_path:
        print(f"loading tokenizer {args.tokenizer_path}")
        return BPETokenizer.from_file(args.tokenizer_path)
    vocab = args.vocab_size or DEFAULT_BPE_VOCAB.get(args.preset, 8192)
    print(f"training byte-level BPE (vocab {vocab}) on {len(files)} file(s)")
    tok = BPETokenizer.train(files, vocab)
    path = Path(args.out).parent / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(path)
    print(f"saved {path}")
    return tok


class WindowSampler:
    """Serves non-overlapping (seq_len + 1)-token windows in a shuffled order, epoch by epoch.

    The batch for a given step is a pure function of (seed, step), so resuming
    from a checkpoint continues the exact same data order without saving RNG state.
    """

    def __init__(self, n_tokens: int, seq_len: int, batch_size: int, seed: int):
        self.seq_len, self.batch_size, self.seed = seq_len, batch_size, seed
        self.n_windows = (n_tokens - 1) // seq_len
        assert self.n_windows > 0, "dataset shorter than one sequence"
        self._epoch, self._perm = -1, None

    def starts(self, step: int) -> list[int]:
        out = []
        for g in range(step * self.batch_size, (step + 1) * self.batch_size):
            epoch, pos = divmod(g, self.n_windows)
            if epoch != self._epoch:
                rng = np.random.default_rng(self.seed + epoch)
                self._epoch, self._perm = epoch, rng.permutation(self.n_windows)
            out.append(int(self._perm[pos]) * self.seq_len)
        return out


def make_batch(data: np.ndarray, starts: list[int], seq_len: int, device: str):
    chunk = torch.from_numpy(np.stack([data[i : i + seq_len + 1] for i in starts]).astype(np.int64))
    if device.startswith("cuda"):
        chunk = chunk.pin_memory().to(device, non_blocking=True)  # overlap the copy with compute
    else:
        chunk = chunk.to(device)
    return chunk[:, :-1], chunk[:, 1:]


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, args) -> float:
    """Loss on a fixed, evenly spaced set of validation windows (comparable across evals)."""
    model.eval()
    n = args.eval_iters * args.batch_size
    starts = np.linspace(0, len(data) - args.seq_len - 2, n).astype(int).tolist()
    losses = []
    for i in range(0, n, args.batch_size):
        with autocast(args.device, args.precision):
            out = model(*make_batch(data, starts[i : i + args.batch_size], args.seq_len, args.device))
        losses.append(out.metrics["lm_loss"])
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: str, model, opt, cfg, tokenizer, step: int) -> None:
    """Write atomically so a crash mid-save never corrupts the previous checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(
        {"config": cfg.to_dict(), "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
         "tokenizer": {"kind": tokenizer.kind, "serialized": tokenizer.to_str()}},
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any ModelConfig field")
    ap.add_argument("--data", nargs="+", default=None, help="text file(s) (default: tiny shakespeare)")
    ap.add_argument("--data-dir", default=None, help="pre-tokenized dir from prepare_data.py")
    ap.add_argument("--max-train-tokens", type=float, default=None,
                    help="use only the first N tokens of train.bin (e.g. 20 x params for a smaller preset)")
    ap.add_argument("--tokenizer", choices=["bpe", "bytes"], default="bpe")
    ap.add_argument("--vocab-size", type=int, default=None, help="BPE vocab (default: per preset)")
    ap.add_argument("--tokenizer-path", default=None, help="reuse a saved tokenizer.json instead of training")
    ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--steps", type=int, default=None, help="default: one pass over --data-dir, else 2000")
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--seq-len", type=int, default=None, help="default: max_seq_len")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    ap.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                    help="auto = bf16 autocast on CUDA, fp32 elsewhere")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA; needs torch >= 2.4)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    args.device = pick_device(args.device)
    args.precision = resolve_precision(args.precision, args.device)
    setup_backends(args.device)
    torch.manual_seed(args.seed)
    if args.data_dir:
        d = Path(args.data_dir)
        tokenizer = BPETokenizer.from_file(d / "tokenizer.json")
        train_data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        if args.max_train_tokens:
            train_data = train_data[: int(args.max_train_tokens)]
        val_data = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
        print(f"{args.data_dir}: {len(train_data) / 1e6:.1f}M train / {len(val_data) / 1e6:.1f}M val tokens")
    else:
        files = resolve_data(args.data)
        tokenizer = build_tokenizer(args, files)
        data = encode_files(tokenizer, files)  # compact uint16 array; batches are widened on the fly
        n_bytes = sum(Path(f).stat().st_size for f in files)
        print(f"{len(data) / 1e6:.2f}M tokens from {n_bytes / 1e6:.2f}MB ({n_bytes / len(data):.2f} bytes/token)")
        split = int(0.9 * len(data))
        train_data, val_data = data[:split], data[split:]

    overrides = parse_overrides(args.set)
    cfg = ModelConfig.from_preset(args.preset, **{**overrides, "vocab_size": tokenizer.vocab_size})
    args.seq_len = args.seq_len or cfg.max_seq_len
    sampler = WindowSampler(len(train_data), args.seq_len, args.batch_size, args.seed)
    if args.steps is None:
        args.steps = sampler.n_windows // (args.batch_size * args.grad_accum) if args.data_dir else 2000
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    print(f"{args.steps:,} steps x {tokens_per_step:,} tokens = {args.steps * tokens_per_step / 1e6:.0f}M tokens"
          f"  |  device {args.device}, precision {args.precision}"
          f"{f', grad-accum {args.grad_accum}' if args.grad_accum > 1 else ''}{', compiled' if args.compile else ''}")

    model = DeepSeekMoEModel(cfg).to(args.device)
    counts = model.param_counts()
    print(f"params: {counts['total'] / 1e6:.2f}M total, {counts['activated'] / 1e6:.2f}M activated/token, "
          f"+{counts['mtp'] / 1e6:.2f}M MTP")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    start = 0
    if args.resume and Path(args.out).exists():
        ckpt = torch.load(args.out, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start = ckpt["step"] + 1
        print(f"resumed from {args.out} at step {start}")

    # Compile a wrapper for the training forward; `model` stays the plain module for saving,
    # evaluation and MoE bias updates (a compiled module's state_dict keys get prefixed).
    fwd = torch.compile(model) if args.compile else model

    t0 = t_log = time.time()
    last_log_step = start
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        opt.zero_grad(set_to_none=True)
        metrics: dict[str, float] = {}
        total_loss = 0.0
        for micro in range(args.grad_accum):
            # Micro-batch index keeps the data order a pure function of (seed, step) for --resume.
            batch = make_batch(train_data, sampler.starts(step * args.grad_accum + micro), args.seq_len, args.device)
            with autocast(args.device, args.precision):
                out = fwd(*batch)
            (out.loss / args.grad_accum).backward()
            total_loss += out.loss.item() / args.grad_accum
            for k, v in out.metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        balance = model.update_moe_biases()  # aux-loss-free load balancing

        if step % args.log_every == 0 or step == args.steps - 1:
            now = time.time()
            m = {**metrics, **balance}
            parts = "  ".join(f"{k} {v:.4f}" for k, v in m.items())
            elapsed = now - t0
            eta = elapsed / (step - start + 1) * (args.steps - step - 1)
            tok_s = (step - last_log_step) * tokens_per_step / max(now - t_log, 1e-9) if step > last_log_step else 0.0
            t_log, last_log_step = now, step
            print(f"step {step:6d}/{args.steps}  loss {total_loss:.4f}  {parts}  lr {lr_at(step, args):.2e}  "
                  f"{(step + 1) * tokens_per_step / 1e6:.1f}M tok  {tok_s:,.0f} tok/s  "
                  f"{elapsed / 3600:.2f}h  eta {eta / 3600:.1f}h", flush=True)
        if (step > 0 and step % args.eval_every == 0) or step == args.steps - 1:
            print(f"  val lm_loss {evaluate(model, val_data, args):.4f}", flush=True)
        if (step > 0 and step % args.save_every == 0) or step == args.steps - 1:
            save_checkpoint(args.out, model, opt, cfg, tokenizer, step)

    print(f"saved {args.out}")
    print(json.dumps(cfg.to_dict(), indent=None))


if __name__ == "__main__":
    main()
