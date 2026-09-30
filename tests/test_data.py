import numpy as np
import pytest
import torch

from mini_transformer.data import get_batch, load_packed
from mini_transformer.tokenizer import decode, encode, load_tokenizer, train_tokenizer

CORPUS = "the quick brown fox jumps over the lazy dog. " * 200


def _packed(tmp_path, block_size=8):
    tok_path = tmp_path / "tokenizer.json"
    train_tokenizer([CORPUS], vocab_size=512, path=tok_path)
    tok = load_tokenizer(tok_path)
    ids = np.array(encode(tok, CORPUS), dtype=np.uint16)
    path = tmp_path / "train.bin"
    path.write_bytes(ids.tobytes())
    return ids, path


def test_batch_shapes_and_offset(tmp_path):
    _, path = _packed(tmp_path)
    data = load_packed(path, block_size=8)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, batch_size=4, block_size=8, device="cpu", generator=g)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_non_ascii_survives_packing(tmp_path):
    tricky = "cafe\u0301 na\u00efve \u201cquotes\u201d \u65e5\u672c\u8a9e"
    tok_path = tmp_path / "tokenizer.json"
    train_tokenizer([CORPUS], vocab_size=512, path=tok_path)
    tok = load_tokenizer(tok_path)
    ids = np.array(encode(tok, tricky), dtype=np.uint16)
    path = tmp_path / "hard.bin"
    path.write_bytes(ids.tobytes())

    data = load_packed(path, block_size=4)
    assert data.dtype == np.uint16
    assert decode(tok, data.tolist()) == tricky


def test_rejects_corpus_smaller_than_block(tmp_path):
    _, path = _packed(tmp_path)
    with pytest.raises(ValueError, match="block_size"):
        load_packed(path, block_size=10_000)


def test_batches_stay_in_bounds(tmp_path):
    _, path = _packed(tmp_path)
    data = load_packed(path, block_size=8)
    g = torch.Generator().manual_seed(1)
    x, _ = get_batch(data, batch_size=32, block_size=8, device="cpu", generator=g)
    assert x.max().item() < len(data)
