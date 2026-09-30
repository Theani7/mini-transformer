import sys
import types

import numpy as np
import pytest
import torch

from mini_transformer.config import Config
from mini_transformer.data import get_batch, load_corpus, load_packed, pack_ids
from mini_transformer.tokenizer import decode, encode, load_tokenizer, train_tokenizer

CORPUS = "the quick brown fox jumps over the lazy dog. " * 200


def _tokenizer(tmp_path):
    tok_path = tmp_path / "tokenizer.json"
    train_tokenizer([CORPUS], vocab_size=512, path=tok_path)
    return load_tokenizer(tok_path)


def _packed(tmp_path, block_size=8):
    tok = _tokenizer(tmp_path)
    ids = np.array(encode(tok, CORPUS), dtype=np.uint16)
    path = tmp_path / "train.bin"
    path.write_bytes(ids.tobytes())
    return ids, path


def _stub_datasets(monkeypatch, rows):
    calls = []

    def fake_load_dataset(dataset, config):
        calls.append((dataset, config))
        return {"train": {"text": rows}}

    stub = types.ModuleType("datasets")
    stub.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", stub)
    return calls


def _assert_windows_are_consecutive(data, x, y, block_size):
    starts = len(data) - block_size
    windows = np.lib.stride_tricks.sliding_window_view(data, block_size)[:starts]
    for row in range(x.shape[0]):
        hits = np.flatnonzero(np.all(windows == x[row].numpy(), axis=1))
        assert hits.size, f"row {row} is not a window of the corpus"
        i = int(hits[0])
        assert 0 <= i and i + block_size + 1 <= len(data)
        assert np.array_equal(data[i + 1 : i + 1 + block_size], y[row].numpy())


def test_batch_shapes_and_offset(tmp_path):
    _, path = _packed(tmp_path)
    data = load_packed(path, block_size=8)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, batch_size=4, block_size=8, device="cpu", generator=g)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])
    _assert_windows_are_consecutive(data, x, y, block_size=8)


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
    ids, path = _packed(tmp_path)
    with pytest.raises(
        ValueError, match=rf"corpus has {len(ids)} tokens but block_size is 10000"
    ):
        load_packed(path, block_size=10_000)


def test_rejects_corpus_one_token_over_the_old_edge(tmp_path):
    block_size = 8
    ids, _ = _packed(tmp_path)
    path = tmp_path / "edge.bin"
    path.write_bytes(ids[: block_size + 1].tobytes())
    with pytest.raises(
        ValueError,
        match=rf"corpus has {block_size + 1} tokens but block_size is {block_size}",
    ):
        load_packed(path, block_size=block_size)


def test_accepts_corpus_at_the_exact_minimum(tmp_path):
    block_size = 8
    ids, _ = _packed(tmp_path)
    path = tmp_path / "min.bin"
    path.write_bytes(ids[: block_size + 2].tobytes())
    data = load_packed(path, block_size=block_size)
    assert len(data) == block_size + 2
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, batch_size=2, block_size=block_size, device="cpu", generator=g)
    assert x.shape == (2, block_size)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batches_stay_in_bounds(tmp_path):
    _, path = _packed(tmp_path)
    data = load_packed(path, block_size=8)
    g = torch.Generator().manual_seed(1)
    x, y = get_batch(data, batch_size=32, block_size=8, device="cpu", generator=g)
    _assert_windows_are_consecutive(data, x, y, block_size=8)
    assert len({tuple(row) for row in x.tolist()}) > 1


def test_default_dataset_id_is_fully_qualified():
    assert Config().dataset == "Salesforce/wikitext"
    assert Config().dataset_config == "wikitext-2-raw-v1"


def test_load_corpus_drops_blank_rows_and_slices(monkeypatch):
    calls = _stub_datasets(monkeypatch, ["alpha", "", "   ", "beta", "gamma"])
    assert load_corpus("Salesforce/wikitext", "wikitext-2-raw-v1", 1000) == (
        "alpha\n\nbeta\n\ngamma"
    )
    assert load_corpus("Salesforce/wikitext", "wikitext-2-raw-v1", 8) == "alpha\n\nb"
    assert calls == [("Salesforce/wikitext", "wikitext-2-raw-v1")] * 2


def test_load_corpus_rejects_blank_only_corpus(monkeypatch):
    calls = _stub_datasets(monkeypatch, ["", "  \n", "\t\t"])
    with pytest.raises(ValueError, match="empty after filtering"):
        load_corpus("Salesforce/wikitext", "wikitext-2-raw-v1", 1000)
    assert calls == [("Salesforce/wikitext", "wikitext-2-raw-v1")]


def test_load_corpus_rejects_zero_chars(monkeypatch):
    _stub_datasets(monkeypatch, ["alpha", "beta"])
    with pytest.raises(ValueError, match="corpus_chars=0"):
        load_corpus("Salesforce/wikitext", "wikitext-2-raw-v1", 0)


def test_pack_ids_writes_uint16_and_returns_count(tmp_path):
    tok = _tokenizer(tmp_path)
    path = tmp_path / "nested" / "packed.bin"
    count = pack_ids(np.array(encode(tok, CORPUS), dtype=np.uint16), path)
    assert count == len(encode(tok, CORPUS))
    assert path.stat().st_size == count * 2
    assert np.frombuffer(path.read_bytes(), dtype=np.uint16).tolist() == (
        np.array(encode(tok, CORPUS), dtype=np.uint16).tolist()
    )
