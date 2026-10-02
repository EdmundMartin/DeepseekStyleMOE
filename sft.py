"""Supervised fine-tuning of a pretrained checkpoint on chat data.

Adds the chat special tokens to the pretrained tokenizer (growing the model's
embedding and output head), then trains only on assistant replies.

Data sources (mix freely):
  --data   JSONL files with {"messages": [{"role": ..., "content": ...}, ...]} per line
           (e.g. the output of gen_sft_data.py)
  --hf     Hugging Face parquet files with a "messages" column, as REPO:PATH, e.g.
           HuggingFaceTB/everyday-conversations-llama3.1-2k:data/train_sft-00000-of-00001.parquet

    python sft.py --init checkpoints/small.pt --data data/fineweb-edu-small/sft_qa.jsonl \\
        --hf HuggingFaceTB/everyday-conversations-llama3.1-2k:data/train_sft-00000-of-00001.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.chat import END, IGNORE, add_chat_tokens, encode_conversation, render_prompt
from deepseek_moe.runtime import autocast, pick_device, resolve_precision, setup_backends
from deepseek_moe.tokenizer import load_tokenizer


def load_conversations(jsonl: list[str], hf: list[str]) -> list[list[dict]]:
    convs = []
    for path in jsonl:
        with open(path) as fh:
            convs += [json.loads(line)["messages"] for line in fh if line.strip()]
    for spec in hf:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        repo, _, file = spec.partition(":")
        local = hf_hub_download(repo, file, repo_type="dataset")
        convs += [row["messages"] for row in pq.read_table(local, columns=["messages"]).to_pylist()]
    return [c for c in convs if c]


def make_batch(examples: list[tuple[list[int], list[int]]], pad_id: int, device: str):
    """Right-pad to the longest example; padding is never trained on.

    Returns (inputs, targets, token_mask); token_mask marks real (non-padding) input positions.
    """
    T = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), T), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), T), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(examples), T), dtype=torch.bool)
    for i, (x, y) in enumerate(examples):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = True
    # Position t predicts token t+1.
    return ids[:, :-1].to(device), labels[:, 1:].to(device), mask[:, :-1].to(device)


@torch.no_grad()
def evaluate(model, val, args) -> float:
    model.eval()
    losses = []
    for i in range(0, len(val), args.batch_size):
        with autocast(args.device, args.precision):
            x, y, mask = make_batch(val[i : i + args.batch_size], args.pad_id, args.device)
            out = model(x, y, token_mask=mask)
        losses.append(out.metrics["lm_loss"])
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample_reply(model, tok, question: str, args) -> str:
    ids = torch.tensor([render_prompt(tok, [{"role": "user", "content": question}])], device=args.device)
    out = model.generate(ids, 120, temperature=0.0, stop_ids={tok.token_to_id(END)})
    model.train()
    return tok.decode(out[0, ids.shape[1] :].tolist()).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="pretrained checkpoint")
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--hf", nargs="*", default=[])
    ap.add_argument("--out", default=None, help="default: <init>-sft.pt")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--device", default="auto", help="auto = cuda > apple-silicon mps > cpu")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                    help="auto = bf16 autocast on CUDA, fp32 elsewhere")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--update-moe-bias", action="store_true",
                    help="keep running aux-loss-free bias updates during SFT (default: frozen at the pretrained "
                         "values, so a small fine-tune can't skew routing that was balanced over pretraining)")
    args = ap.parse_args()
    if not args.data and not args.hf:
        ap.error("give at least one --data or --hf source")

    torch.manual_seed(args.seed)
    args.device = pick_device(args.device)
    args.precision = resolve_precision(args.precision, args.device)
    setup_backends(args.device)
    ckpt = torch.load(args.init, map_location=args.device)
    tok = load_tokenizer(**ckpt["tokenizer"])
    if tok.kind != "bpe":
        raise SystemExit("SFT needs a BPE checkpoint (chat roles are special tokens)")
    model = DeepSeekMoEModel(ModelConfig.from_dict(ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    added = add_chat_tokens(tok)
    model.resize_vocab(tok.vocab_size)
    args.pad_id = tok.eot_id
    print(f"loaded {args.init}; +{added} chat tokens -> vocab {tok.vocab_size}")

    convs = load_conversations(args.data, args.hf)
    examples = [e for c in convs if (e := encode_conversation(tok, c, model.cfg.max_seq_len + 1))]
    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    val, train = examples[:n_val], examples[n_val:]
    trained_tokens = sum(sum(l != IGNORE for l in y) for _, y in train)
    print(f"{len(convs):,} conversations -> {len(examples):,} usable ({len(train):,} train / {len(val):,} val), "
          f"{trained_tokens / 1e6:.2f}M assistant tokens in train")

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, total - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * p))

    probe = next((c[0]["content"] for c in convs[:50] if c[0]["role"] == "user"), "Hello!")
    print(f"val lm_loss before SFT {evaluate(model, val, args):.4f}")
    model.train()
    t0, order = time.time(), []
    for step in range(total):
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        batch = [train[order.pop()] for _ in range(min(args.batch_size, len(order)))]
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with autocast(args.device, args.precision):
            x, y, mask = make_batch(batch, args.pad_id, args.device)
            out = model(x, y, token_mask=mask)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        balance = model.update_moe_biases(update=args.update_moe_bias)  # frozen by default; stats only

        if step % args.log_every == 0 or step == total - 1:
            m = "  ".join(f"{k} {v:.4f}" for k, v in {**out.metrics, **balance}.items())
            print(f"step {step:5d}/{total}  {m}  lr {lr_at(step):.2e}  {time.time() - t0:.0f}s", flush=True)
        if (step > 0 and step % args.eval_every == 0) or step == total - 1:
            print(f"  val lm_loss {evaluate(model, val, args):.4f}")
            print(f"  Q: {probe}\n  A: {sample_reply(model, tok, probe, args)}", flush=True)

    out_path = args.out or str(Path(args.init).with_name(Path(args.init).stem + "-sft.pt"))
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict(), "chat": True,
                "tokenizer": {"kind": tok.kind, "serialized": tok.to_str()}}, out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
