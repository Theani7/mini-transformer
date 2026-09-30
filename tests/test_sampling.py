import torch

from mini_transformer.sampling import sample_next


def test_returns_vocab_ids_in_range():
    logits = torch.randn(3, 500)
    out = sample_next(logits, temperature=0.8, top_p=0.9)
    assert out.shape == (3, 1)
    assert out.min() >= 0
    assert out.max() < 500


def test_tight_top_p_equals_argmax():
    """A distribution with one dominant token must sample that token.

    This is the regression test for measurements.md 5a: without the
    permutation remap this returns a sorted-array index instead.
    """
    logits = torch.full((1, 1000), -20.0)
    logits[0, 731] = 20.0
    out = sample_next(logits, temperature=1.0, top_p=0.9)
    assert out.item() == 731


def test_zero_temperature_is_greedy():
    logits = torch.randn(2, 100)
    expected = logits.argmax(dim=-1, keepdim=True)
    assert torch.equal(sample_next(logits, temperature=0.0), expected)


def test_high_temperature_stays_in_vocab():
    torch.manual_seed(0)
    logits = torch.randn(4, 256)
    for out in sample_next(logits, temperature=5.0, top_p=1.0):
        assert 0 <= out.item() < 256


def test_top_p_1_is_plain_sampling():
    torch.manual_seed(0)
    logits = torch.randn(1, 50)
    a = sample_next(logits, temperature=1.0, top_p=1.0)
    b = sample_next(logits, temperature=1.0, top_p=1.0)
    assert a.shape == b.shape == (1, 1)
