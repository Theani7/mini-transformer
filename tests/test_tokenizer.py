import pytest

from mini_transformer.tokenizer import (
    decode,
    encode,
    load_tokenizer,
    train_tokenizer,
)

CORPUS = "the quick brown fox jumps over the lazy dog. " * 40


def test_trains_and_roundtrips(tmp_path):
    path = tmp_path / "tokenizer.json"
    size = train_tokenizer([CORPUS], vocab_size=512, path=path)
    assert path.exists()
    assert size > 0
    tok = load_tokenizer(path)
    assert decode(tok, encode(tok, CORPUS)) == CORPUS


def test_roundtrips_non_ascii(tmp_path):
    path = tmp_path / "tokenizer.json"
    train_tokenizer([CORPUS], vocab_size=512, path=path)
    tok = load_tokenizer(path)
    tricky = "café — naïve “quotes” 日本語 🎉"
    assert decode(tok, encode(tok, tricky)) == tricky


def test_rejects_vocab_above_uint16(tmp_path):
    with pytest.raises(ValueError, match="65535"):
        train_tokenizer([CORPUS], vocab_size=70000, path=tmp_path / "t.json")


def test_rejects_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        train_tokenizer(["   \n  "], vocab_size=512, path=tmp_path / "t.json")


def test_warns_when_vocab_falls_short_of_request(tmp_path, capsys):
    # A short corpus cannot supply enough merges; 4096 is unreachable here.
    path = tmp_path / "t.json"
    size = train_tokenizer(["short text"], vocab_size=4096, path=path)
    tok = load_tokenizer(path)
    assert "4096" in capsys.readouterr().out
    assert size == tok.get_vocab_size() < 4096
