"""Tokenizers: a byte-level BPE trained on your own corpus (the default), or raw bytes.

The BPE is byte-level, like DeepSeek-V3's: its base alphabet is all 256 bytes,
so any input encodes without unknown tokens, and merges are learned from the
training corpus. Both tokenizers serialize to a string so a checkpoint can
carry its own tokenizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"

# Default BPE vocab per preset. The embedding and output head each cost vocab * dim
# params, so small models need small vocabularies.
DEFAULT_BPE_VOCAB = {"tiny": 8192, "mini": 8192, "small": 8192, "medium": 8192, "base": 32768,
                     "large": 32768, "xl": 65536, "v2-lite": 102400, "v3": 129280}


class BPETokenizer:
    kind = "bpe"

    def __init__(self, tok: Tokenizer):
        self.tok = tok

    @staticmethod
    def _new(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[EOT],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )
        return tok, trainer

    @classmethod
    def train(cls, files: list[str], vocab_size: int) -> "BPETokenizer":
        tok, trainer = cls._new(vocab_size)
        tok.train(files, trainer)
        return cls(tok)

    @classmethod
    def train_from_texts(cls, texts: Iterable[str], vocab_size: int) -> "BPETokenizer":
        tok, trainer = cls._new(vocab_size)
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @classmethod
    def from_str(cls, s: str) -> "BPETokenizer":
        return cls(Tokenizer.from_str(s))

    @classmethod
    def from_file(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def to_str(self) -> str:
        return self.tok.to_str()

    def save(self, path: str | Path) -> None:
        self.tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return self.tok.token_to_id(EOT)

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Register special tokens (e.g. chat roles); returns how many were new."""
        return self.tok.add_special_tokens(tokens)

    def token_to_id(self, token: str) -> int | None:
        return self.tok.token_to_id(token)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


class ByteTokenizer:
    """One token per byte (vocab 256); handy for quick experiments."""

    kind = "bytes"
    vocab_size = 256
    eot_id = None

    def to_str(self) -> str:
        return ""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def load_tokenizer(kind: str, serialized: str = "") -> BPETokenizer | ByteTokenizer:
    return BPETokenizer.from_str(serialized) if kind == "bpe" else ByteTokenizer()


def _chunks(files: Iterable[str], lines_per_chunk: int = 10_000) -> Iterator[list[str]]:
    """Yield batches of lines (newlines kept) so large corpora never sit in memory as one string."""
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            batch = []
            for line in fh:
                batch.append(line)
                if len(batch) == lines_per_chunk:
                    yield batch
                    batch = []
            if batch:
                yield batch


def encode_files(tokenizer: BPETokenizer | ByteTokenizer, files: list[str]) -> np.ndarray:
    """Encode every file into one flat token array, with EOT between documents when available."""
    dtype = np.uint16 if tokenizer.vocab_size <= 65536 else np.int32
    parts: list[np.ndarray] = []
    for i, f in enumerate(files):
        if i > 0 and tokenizer.eot_id is not None:
            parts.append(np.array([tokenizer.eot_id], dtype=dtype))
        for batch in _chunks([f]):
            ids = tokenizer.encode_batch(batch)
            parts.append(np.fromiter((t for seq in ids for t in seq), dtype=dtype))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
