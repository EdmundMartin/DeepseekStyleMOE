# deepseek-moe-style

A small DeepSeek-V3-style MoE language model in plain PyTorch. You can set its size.

| Component | Where | What it does |
|---|---|---|
| Multi-head Latent Attention | `deepseek_moe/mla.py` | Caches only a low-rank latent `c_KV` and one shared RoPE key per token. Training uses the naive up-projection. Cached decoding uses the weight-absorbed path. |
| DeepSeekMoE | `deepseek_moe/moe.py` | Shared experts plus fine-grained routed experts. Uses sigmoid gating, group-limited routing, aux-loss-free bias balancing and a small sequence-wise balance loss. |
| Multi-Token Prediction | `deepseek_moe/model.py` (`MTPModule`) | D sequential MTP depths that share the embedding and output head. |
| FP8 | `deepseek_moe/fp8.py` | Simulated E4M3 with 1×128 activation tiles and 128×128 weight blocks during training. `freeze_fp8()` stores weights as real float8. |

## Setup

PyTorch 2.2.2 is the last release with Intel-Mac wheels, so the project is pinned to Python 3.12:

```sh
uv venv --python 3.12 venv && VIRTUAL_ENV=venv uv sync --active
```

## Tokenizer

By default `train.py` first trains a byte-level BPE on the corpus you give it, using Hugging Face `tokenizers`. It saves the tokenizer to `checkpoints/tokenizer.json` and stores it inside the checkpoint. The default vocab size depends on the preset: 2K (`tiny`), 8K (`small`/`medium`) and 32K (`base`). Change it with `--vocab-size`. Use `--tokenizer-path` to reuse a trained tokenizer, or `--tokenizer bytes` for raw bytes.

## Usage

```sh
venv/bin/python train.py --preset tiny --steps 2000          # tiny shakespeare, CPU-friendly
venv/bin/python train.py --data corpus/*.txt --preset small --vocab-size 16000
venv/bin/python train.py --preset small --set use_fp8=true n_mtp_modules=2
venv/bin/python sample.py --ckpt checkpoints/small.pt       # interactive: /top, /ppl, /set, /info
venv/bin/python sample.py --prompt "ROMEO:" --tokens 400    # one-shot
venv/bin/python -m pytest tests
```

### Chinchilla-sized FineWeb-Edu run

`prepare_data.py` sizes the dataset at 20 tokens per parameter (the Chinchilla ratio) for a preset. It downloads FineWeb-Edu `sample/10BT` Parquet files, trains the BPE and writes `train.bin` and `val.bin` as uint16 token IDs. `train.py --data-dir` then makes one shuffled pass over the data. It checkpoints every `--save-every` steps, and `--resume` continues from the last checkpoint. Keep `--steps` and `--batch-size` the same when resuming, so the learning-rate schedule and data order stay the same.

```sh
venv/bin/python prepare_data.py --preset small      # 286M tokens, ~1.1GB of text
caffeinate -i venv/bin/python train.py --data-dir data/fineweb-edu-small --preset small \
    --batch-size 8 --lr 1e-3 --min-lr 1e-4 --warmup 700 --out checkpoints/small.pt [--resume]
```

### SFT (chat fine-tuning)

1. **Generate grounded Q&A.** `gen_sft_data.py` reads documents back out of `train.bin`, so every question is about text the model saw during pretraining. It asks a larger teacher model behind any OpenAI-compatible endpoint (vLLM, Ollama, LM Studio, OpenRouter and so on) to write question/answer pairs. Pairs that mention "the document" are dropped. Re-running with the same `--out` resumes where it stopped.
2. **Fine-tune.** `sft.py` adds `<|system|> <|user|> <|assistant|> <|end|>` tokens, grows the embedding and output layers, and trains only on assistant replies and the `<|end|>` that ends them. You can mix JSONL files with Hugging Face parquet files that have a `messages` column.
3. **Chat.** Run `chat.py`.

```sh
venv/bin/python gen_sft_data.py --base-url http://localhost:11434/v1 --model <teacher> --docs 5000
venv/bin/python sft.py --init checkpoints/small.pt --data data/fineweb-edu-small/sft_qa.jsonl \
    --hf HuggingFaceTB/everyday-conversations-llama3.1-2k:data/train_sft-00000-of-00001.parquet
venv/bin/python chat.py --ckpt checkpoints/small-sft.pt
```

### Export to GGUF (llama.cpp, Ollama, LM Studio)

`export_gguf.py` writes a checkpoint straight to GGUF in llama.cpp's native `deepseek2` architecture. It covers MLA with the absorbed `k_b`/`v_b` split, sigmoid-gated MoE with shared experts, expert groups and the balancing bias. The MTP module is training-only and is left out. Base checkpoints get `<|endoftext|>` as BOS/EOS. SFT checkpoints also get the chat template, with `<|end|>` as end of turn.

```sh
venv/bin/python export_gguf.py --ckpt checkpoints/mini.pt --dtype f16          # or f32 / q8_0
llama-completion -m checkpoints/mini-f16.gguf -p "The history of" -n 100
llama-cli -m checkpoints/mini-sft-q8_0.gguf                                     # chat, uses the embedded template
```

It was checked against llama.cpp b11203. Tokenization is identical, and greedy generation matches our PyTorch model token for token in f32, for both base and chat checkpoints. q8_0 differs only where the top two tokens are nearly tied.

Presets with their default BPE vocabularies are `tiny` (3.0M params, 2.6M active per token), `mini` (11.0M, 6.3M active; deep-and-thin, tuned for quality), `small` (14.3M, 8.4M active), `medium` (85.9M, 30.8M active) and `base` (283.9M, 130.5M active). You can override any `ModelConfig` field with `--set key=value`.

For reference only (too big for a laptop), there are also `large` (1.31B, 0.32B active), `xl` (5.83B, 1.11B active), `v2-lite` (DeepSeek-V2-Lite's shape: 15.7B, 2.4B active) and `v3` (DeepSeek-V3's exact configuration: 671B, 37B active). `ModelConfig.param_estimate()` computes the exact sizes without building the model, and it reproduces DeepSeek's published figures for V2-Lite and V3.
