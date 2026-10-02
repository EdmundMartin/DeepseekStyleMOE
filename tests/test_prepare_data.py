import pyarrow as pa
import pytest

from deepseek_moe.chat import CHAT_TOKENS, add_chat_tokens
from deepseek_moe.tokenizer import BPETokenizer
from prepare_data import _code_rows, _render_chat, parse_mix


def test_parse_mix_normalises_and_rejects_unknown():
    assert parse_mix(["fineweb-edu=3", "code=1"]) == {"fineweb-edu": 0.75, "code": 0.25}
    with pytest.raises(SystemExit):
        parse_mix(["nope=1"])


def test_render_chat_uses_chat_template_tokens():
    text = _render_chat([{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                         {"role": "assistant", "content": "A"}, {"role": "tool", "content": "ignored"}])
    assert text == "<|system|>S<|end|><|user|>Q<|end|><|assistant|>A<|end|>"


def test_code_filter_keeps_permissive_known_languages_only():
    batch = pa.table({"code": ["ok", "gpl", "lang", "huge"], "language": ["Python", "Python", "Brainfuck", "Go"],
                      "license": ["mit", "gpl-3.0", "mit", "apache-2.0"], "size": [10, 10, 10, 10_000_000]})
    assert list(_code_rows(batch.to_batches()[0])) == ["ok"]


def test_chat_tokens_trained_in_mean_no_resize_for_sft():
    tok = BPETokenizer.train_from_texts(["hello world " * 50], 300, extra_special=CHAT_TOKENS)
    before = tok.vocab_size
    assert add_chat_tokens(tok) == 0 and tok.vocab_size == before
    ids = tok.encode("<|user|>hi<|end|>")
    assert ids[0] == tok.token_to_id("<|user|>") and ids[-1] == tok.token_to_id("<|end|>")
