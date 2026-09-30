import torch

from mini_transformer.sampling import sample_next


def test_returns_vocab_ids_in_range():
    torch.manual_seed(0)
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


def test_sampling_is_actually_stochastic():
    torch.manual_seed(0)
    logits = torch.randn(1, 64)
    seen = {sample_next(logits, temperature=1.0, top_p=1.0).item() for _ in range(20)}
    assert len(seen) > 1


def test_top_p_truncates_the_tail():
    torch.manual_seed(0)
    logits = torch.tensor([[4.0, 3.0, 2.0, 0.0]])  # strictly decreasing: no tie ambiguity
    seen = {sample_next(logits, temperature=1.0, top_p=0.75).item() for _ in range(50)}
    assert seen == {0, 1}


def test_top_k_restricts_to_top_candidates():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
    seen = {sample_next(logits, temperature=1.0, top_k=2).item() for _ in range(30)}
    assert seen == {0, 1}


def test_repetition_penalty_depresses_seen_tokens():
    logits = torch.tensor([[5.0, 4.0]])
    context = torch.tensor([[0]])
    # Token 0 is preferred (5.0 > 4.0). With penalty 2.0, 5.0 / 2.0 = 2.5 < 4.0, so token 1 wins.
    out = sample_next(
        logits,
        temperature=0.0,
        repetition_penalty=2.0,
        context=context,
    )
    assert out.item() == 1
