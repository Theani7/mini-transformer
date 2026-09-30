import numpy as np
import pytest
import torch

from mini_transformer.data import split_ids
from mini_transformer.train import evaluate


def test_split_is_sequential_disjoint_and_exact():
    ids = np.arange(10_000, dtype=np.uint16)

    train_ids, val_ids = split_ids(ids, val_fraction=0.05, block_size=128)

    assert len(val_ids) == 500
    assert len(train_ids) == 9_500
    assert not set(train_ids.tolist()) & set(val_ids.tolist())
    # sequential: val is the tail, not a random sample
    assert train_ids[-1] == ids[9_499]
    assert val_ids[0] == ids[9_500]


def test_split_rounds_by_documented_boundary():
    ids = np.arange(1001, dtype=np.uint16)
    train_ids, val_ids = split_ids(ids, val_fraction=0.05, block_size=8)
    # 1001 * 0.05 = 50.05 -> 50 held out, 951 trained on
    assert len(val_ids) == 50
    assert len(train_ids) == 951


def test_split_rejects_a_fraction_that_would_starve_the_val_set():
    ids = np.arange(100, dtype=np.uint16)
    with pytest.raises(ValueError, match="val"):
        split_ids(ids, val_fraction=0.0, block_size=128)


def test_split_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="empty"):
        split_ids(np.array([], dtype=np.uint16), val_fraction=0.05, block_size=8)


def _model():
    from mini_transformer.config import Config
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    torch.manual_seed(0)
    model = MiniTransformer(vocab_size=64, config=Config())
    # Without this the fixture keeps the default N(0,1) embedding init and a
    # loss near 175 - the 182-vs-9.03 bug from measurements.md section 3.
    init_weights(model, Config().n_layer)
    return model.eval()


def test_evaluate_returns_the_mean_loss_over_the_whole_split():
    model = _model().eval()
    data = np.arange(200, dtype=np.uint16) % 64

    loss_a = evaluate(model, data, batch_size=4, block_size=16, device="cpu")
    loss_b = evaluate(model, data, batch_size=4, block_size=16, device="cpu")
    loss_c = evaluate(model, data, batch_size=8, block_size=16, device="cpu")

    assert loss_a == pytest.approx(loss_b)
    # every window is covered either way, so batch size must not change the mean
    assert loss_a == pytest.approx(loss_c, rel=0.05)


def test_evaluate_does_not_train_or_mutate_weights():
    model = _model().eval()
    data = np.arange(200, dtype=np.uint16) % 64
    before = model.token_embedding.weight.detach().clone()

    evaluate(model, data, batch_size=4, block_size=16, device="cpu")

    assert torch.equal(before, model.token_embedding.weight)
    assert model.training is False


def test_evaluate_needs_no_seed_because_coverage_is_exhaustive():
    """Every window is scored, so the result is exactly reproducible."""
    model = _model().eval()
    data = np.arange(200, dtype=np.uint16) % 64

    first = evaluate(model, data, 4, 16, "cpu")
    second = evaluate(model, data, 4, 16, "cpu")

    assert first == pytest.approx(second)
    assert first == pytest.approx(evaluate(model, data, 3, 16, "cpu"), rel=0.05)

def test_evaluate_is_a_token_mean_not_a_sum_per_window():
    """Regression: dividing the summed token loss by the *window* count
    instead of the token count inflates the result by block_size. A real run
    reported val loss 819.976 where 6.406 was correct."""
    import torch.nn.functional as F

    from mini_transformer.rope import rope_cache

    model = _model().eval()
    data = np.arange(400, dtype=np.uint16) % 64
    block_size = 16

    got = evaluate(model, data, 8, block_size, "cpu")

    cos, sin = rope_cache(block_size, model.config.head_dim)
    windows = len(data) - block_size - 1
    total = 0.0
    with torch.no_grad():
        for i in range(windows):
            x = torch.from_numpy(data[i : i + block_size].astype(np.int64))[None]
            y = torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))[None]
            total += F.cross_entropy(
                model(x, cos, sin)[0], y[0], reduction="sum"
            ).item()

    assert got == pytest.approx(total / (windows * block_size), rel=1e-5)
    assert got < 8.0
