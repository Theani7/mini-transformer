import torch

from mini_transformer.config import Config
from mini_transformer.model import MiniTransformer
from mini_transformer.multi_head_attention import MultiHeadAttention
from mini_transformer.rope import rope_cache

CONFIG = Config()
VOCAB = 128


def _model():
    torch.manual_seed(0)
    return MiniTransformer(vocab_size=VOCAB, config=CONFIG).eval()


def test_greedy_generation_is_identical_with_and_without_the_cache():
    """The load-bearing property: the cache is an optimisation, not a behaviour change.

    Any divergence is either RoPE positions read chunk-relative, or a causal
    mask applied at decode where it must not be.
    """
    model = _model()
    tokens = torch.randint(0, VOCAB, (1, 12))

    cached = model.generate(tokens, 24, config=CONFIG, temperature=0.0, use_cache=True)
    plain = model.generate(tokens, 24, config=CONFIG, temperature=0.0, use_cache=False)

    assert torch.equal(cached, plain)


def test_cached_decode_logits_match_an_uncached_full_prefix():
    """Localises which half broke: the decode step at position p must equal
    the last position of an uncached forward over the whole prefix."""
    model = _model()
    prefix = torch.randint(0, VOCAB, (1, 10))

    with torch.no_grad():
        cache = [{} for _ in model.blocks]
        cos, sin = rope_cache(CONFIG.block_size, CONFIG.head_dim)

        model(prefix[:, :-1], cos, sin, cache)
        cached_logits = model(prefix[:, -1:], cos, sin, cache)

        full_logits = model(prefix, cos, sin)

    assert torch.allclose(cached_logits[0, -1], full_logits[0, -1], atol=1e-5)


def test_decode_attends_to_every_cached_key_not_just_the_first():
    """is_causal=True aligns top-left, so at decode (q_len=1, k_len=N) it would
    let the query see only key 0. A decode query must see all N."""
    torch.manual_seed(0)
    d_model, n_head, seq_len = 32, 4, 16
    head_dim = d_model // n_head

    attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
    cos, sin = rope_cache(seq_len, head_dim)

    x = torch.randn(1, seq_len, d_model)
    with torch.no_grad():
        cache = {}
        attn(x[:, :-1], cos, sin, cache)
        got = attn(x[:, -1:], cos[:, :, seq_len - 1 :], sin[:, :, seq_len - 1 :], cache)

        # Reference: recompute the whole prefix and take its last position.
        expected = attn(x, cos, sin)[:, -1:]

    assert torch.allclose(got, expected, atol=1e-5)
    assert cache["k"].shape[2] == seq_len


def test_the_cache_grows_by_one_position_per_step():
    model = _model()
    tokens = torch.randint(0, VOCAB, (1, 5))
    cos, sin = rope_cache(CONFIG.block_size, CONFIG.head_dim)

    cache = [{} for _ in model.blocks]
    with torch.no_grad():
        model(tokens, cos, sin, cache)
        assert cache[0]["k"].shape[2] == 5

        model(tokens[:, -1:], cos, sin, cache)
        assert cache[0]["k"].shape[2] == 6


def test_no_cache_means_no_side_effects_and_unchanged_shapes():
    model = _model()
    tokens = torch.randint(0, VOCAB, (2, 16))
    cos, sin = rope_cache(CONFIG.block_size, CONFIG.head_dim)

    with torch.no_grad():
        assert model(tokens, cos, sin).shape == (2, 16, VOCAB)
        assert model(tokens, cos, sin, cache=None).shape == (2, 16, VOCAB)


def test_uncached_generation_preserves_context_across_steps():
    """Uncached generation used `window = next_token` which dropped all past context
    from step 2 onward, turning generation into a 1-token Markov model. Plain and
    cached generation must match on perturbed weights where argmax does not trivially loop."""
    model = _model()
    for p in model.parameters():
        p.data.normal_(0, 0.5)

    tokens = torch.randint(0, VOCAB, (1, 12))
    cached = model.generate(tokens, 10, config=CONFIG, temperature=0.0, use_cache=True)
    plain = model.generate(tokens, 10, config=CONFIG, temperature=0.0, use_cache=False)

    assert torch.equal(cached, plain)