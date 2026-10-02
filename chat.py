"""Chat with an SFT checkpoint in the terminal.

    python chat.py --ckpt checkpoints/small-sft.pt
    python chat.py --system "You are a concise tutor." --temperature 0.7

Commands: /reset clears the conversation, /quit exits.
"""

from __future__ import annotations

import argparse

import torch

from deepseek_moe import DeepSeekMoEModel, ModelConfig
from deepseek_moe.chat import END, render_prompt
from deepseek_moe.runtime import pick_device
from deepseek_moe.tokenizer import load_tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/mini-sft.pt")
    ap.add_argument("--system", default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--rep-penalty", type=float, default=1.1, help="repetition penalty; 1.0 disables")
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--device", default="auto", help="auto = cuda > apple-silicon mps > cpu")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    if not ckpt.get("chat"):
        raise SystemExit(f"{args.ckpt} is not an SFT checkpoint; run sft.py first")
    tok = load_tokenizer(**ckpt["tokenizer"])
    model = DeepSeekMoEModel(ModelConfig.from_dict(ckpt["config"])).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    end_id = tok.token_to_id(END)

    history: list[dict] = [{"role": "system", "content": args.system}] if args.system else []
    print("chat ready (/reset, /quit)")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/quit":
            break
        if user == "/reset":
            history = history[:1] if args.system else []
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        prompt = render_prompt(tok, history)
        # Keep the most recent context that fits, leaving room for the reply.
        prompt = prompt[-(model.cfg.max_seq_len - args.max_new_tokens):]
        ids = torch.tensor([prompt], device=args.device)
        out = model.generate(ids, args.max_new_tokens, temperature=args.temperature,
                             top_k=args.top_k, top_p=args.top_p,
                             repetition_penalty=args.rep_penalty, stop_ids={end_id})
        reply_ids = [t for t in out[0, ids.shape[1]:].tolist() if t != end_id]
        reply = tok.decode(reply_ids).strip()
        print(f"bot> {reply}")
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
