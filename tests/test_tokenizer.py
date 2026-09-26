from deepseek_moe.tokenizer import BPETokenizer, ByteTokenizer, encode_files, load_tokenizer


def test_bpe_trains_roundtrips_and_serializes(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("the quick brown fox jumps over the lazy dog\n" * 200)
    b.write_text("héllo wörld — ünïcode 🙂\n" * 50)
    tok = BPETokenizer.train([str(a), str(b)], vocab_size=400)
    assert 256 < tok.vocab_size <= 400

    text = "the lazy fox says héllo 🙂\n  indented"
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("the quick brown fox")) < len("the quick brown fox")  # merges learned

    restored = load_tokenizer("bpe", tok.to_str())
    assert restored.encode(text) == tok.encode(text)

    ids = encode_files(tok, [str(a), str(b)])
    assert (ids == tok.eot_id).sum() == 1  # one separator between the two documents
    assert ids.max() < tok.vocab_size


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    assert tok.decode(tok.encode("héllo")) == "héllo"
