import pytest
import torch
import torch.nn.functional as F

from mini_transformer.feed_forward import SwiGLU
from mini_transformer.multi_head_attention import MultiHeadAttention
from mini_transformer.rmsnorm import RMSNorm
from mini_transformer.rope import apply_rope, rope_cache, rotate_half


def test_rmsnorm_output_has_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(2, 4, 64) * 5
    out = norm(x)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_respects_learned_scale():
    norm = RMSNorm(8)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    out = norm(torch.randn(1, 8))
    assert out.pow(2).mean().sqrt().item() == pytest.approx(2.0, abs=1e-3)


def test_rmsnorm_eps_guards_zero_input():
    norm = RMSNorm(8)
    out = norm(torch.zeros(1, 8))
    assert torch.isfinite(out).all()
    assert out.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_rope_cache_shape_and_odd_dim_rejected():
    cos, sin = rope_cache(16, 8)
    assert cos.shape == (1, 1, 16, 8)
    assert sin.shape == (1, 1, 16, 8)
    with pytest.raises(ValueError, match="even"):
        rope_cache(16, 7)


def test_rotate_half_swaps_and_negates():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_depends_only_on_relative_offset():
    torch.manual_seed(0)
    cos, sin = rope_cache(64, 8)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    first = apply_rope(q, cos[:, :, 5:6], sin[:, :, 5:6]) @ apply_rope(
        k, cos[:, :, 11:12], sin[:, :, 11:12]
    ).mT
    second = apply_rope(q, cos[:, :, 20:21], sin[:, :, 20:21]) @ apply_rope(
        k, cos[:, :, 26:27], sin[:, :, 26:27]
    ).mT
    assert torch.allclose(first, second, atol=1e-5)


# GPT-NeoX reference, head_dim=8, base=1e4, at position 3: theta_i = 3 * 1e4**(-i/4)
# for i = 0..3, giving theta = (3, 0.3, 0.03, 0.003). Probing every pair is what
# pins base and the frequency ladder; pair 0 alone is blind to base.
NEOX_LADDER_AT_3 = (
    (-0.9899924966, 0.1411200081),
    (0.9553364891, 0.2955202067),
    (0.9995500337, 0.0299955002),
    (0.9999955000, 0.0029999955),
)


def test_rope_matches_neox_rotation():
    cos, sin = rope_cache(8, 8)
    e0 = torch.tensor([[[[1.0, 0, 0, 0, 0, 0, 0, 0]]]])
    out = apply_rope(e0, cos[:, :, 1:2], sin[:, :, 1:2]).flatten()
    assert torch.allclose(out, torch.tensor([0.5403, 0, 0, 0, 0.8415, 0, 0, 0]), atol=1e-4)
    for pair, (expected_cos, expected_sin) in enumerate(NEOX_LADDER_AT_3):
        e = torch.zeros(1, 1, 1, 8)
        e[0, 0, 0, pair] = 1.0
        rotated = apply_rope(e, cos[:, :, 3:4], sin[:, :, 3:4]).flatten()
        assert rotated[pair] == pytest.approx(expected_cos, abs=1e-6)
        assert rotated[pair + 4] == pytest.approx(expected_sin, abs=1e-6)


def test_rope_broadcasts_over_batch_and_heads():
    cos, sin = rope_cache(6, 8)
    x = torch.randn(2, 4, 6, 8)
    out = apply_rope(x, cos[:, :, :6], sin[:, :, :6])
    assert out.shape == (2, 4, 6, 8)
    per_position = torch.cat(
        [
            apply_rope(x[:, :, t : t + 1], cos[:, :, t : t + 1], sin[:, :, t : t + 1])
            for t in range(6)
        ],
        dim=2,
    )
    assert torch.allclose(out, per_position)


def test_swiglu_shape_gating_and_no_projection_biases():
    ffn = SwiGLU(16, 32)
    x = torch.randn(2, 3, 16)
    assert ffn(x).shape == (2, 3, 16)
    assert ffn.gate.bias is None and ffn.up.bias is None and ffn.down.bias is None
    # gate is multiplied by up, so zeroing up must zero the output
    with torch.no_grad():
        ffn.gate.weight.zero_()
    assert torch.allclose(ffn(x), torch.zeros(2, 3, 16))
    # restore the gate so the up probe is not vacuous, then zero up instead
    with torch.no_grad():
        ffn.gate.weight.normal_()
        ffn.up.weight.zero_()
    assert torch.allclose(ffn(x), torch.zeros(2, 3, 16))


def test_swiglu_is_silu_of_gate_times_up_pushed_through_down():
    """The only value-level check in the model, and the only one that can fail
    for the arithmetic itself.

    Zeroing each projection in turn is symmetric under swapping `gate` and `up`
    (both produce an all-zero output) and completely blind to which activation
    multiplies the gate, so both mutants survive it. Here every weight is
    hand-set and every input is 1.0, so the product is read straight off:

        p0 = silu(1.0) * 3.0 = 2.1931757      p1 = silu(0.5) * 2.0 = 0.6224593
        out = 1.0 * p0 - 1.0 * p1 = 1.5707164

    Swapping gate and up gives 1.9769253; silu->gelu gives 1.8325715.
    """
    ffn = SwiGLU(1, 2)
    with torch.no_grad():
        ffn.gate.weight.copy_(torch.tensor([[1.0], [0.5]]))
        ffn.up.weight.copy_(torch.tensor([[3.0], [2.0]]))
        ffn.down.weight.copy_(torch.tensor([[1.0, -1.0]]))

    assert ffn(torch.ones(1, 1, 1)).item() == pytest.approx(1.5707164, abs=1e-6)


def _attention(block_size=16, d_model=32, n_head=4):
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=d_model, n_head=n_head)
    cos, sin = rope_cache(block_size, d_model // n_head)
    return attn, cos, sin


def test_attention_preserves_shape():
    attn, cos, sin = _attention()
    x = torch.randn(2, 16, 32)
    assert attn(x, cos, sin).shape == (2, 16, 32)


def test_attention_is_causal():
    """No output position may depend on a later input position."""
    attn, cos, sin = _attention()
    x = torch.randn(1, 16, 32)
    before = attn(x, cos, sin)

    perturbed = x.clone()
    perturbed[0, 8:] = torch.randn(8, 32) * 5
    after = attn(perturbed, cos, sin)

    assert torch.allclose(before[0, :8], after[0, :8], atol=1e-6)
    assert not torch.allclose(before[0, 8:], after[0, 8:], atol=1e-6)


def test_attention_rejects_indivisible_head_count():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=30, n_head=4)


def test_attention_rots_q_and_k_but_not_v():
    """Recompute the intended wiring from the module's own qkv weights.

    Every other attention test is invariant to which of q/k/v gets rotated, and
    to the argument order handed to sdpa, so this is the only thing pinning it.
    """
    attn, cos, sin = _attention()
    x = torch.randn(2, 16, 32)
    batch_size, seq_len, d_model = x.shape
    cos, sin = cos[:, :, :seq_len], sin[:, :, :seq_len]

    def split(t):
        return t.view(batch_size, seq_len, attn.n_head, attn.head_dim).transpose(1, 2)

    q, k, v = (split(t) for t in attn.qkv(x).split(d_model, dim=2))

    expected = F.scaled_dot_product_attention(
        apply_rope(q, cos, sin),
        apply_rope(k, cos, sin),
        v,
        is_causal=True,
    )
    expected = attn.proj(
        expected.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    )

    assert torch.allclose(attn(x, cos, sin), expected, atol=1e-6)


def test_apply_rope_rejects_non_4d_input():
    cos, sin = rope_cache(8, 8)
    with pytest.raises(ValueError, match="batch, heads, seq, head_dim"):
        apply_rope(torch.randn(2, 8, 8), cos, sin)


def test_apply_rope_rejects_mismatched_head_dim():
    cos, sin = rope_cache(8, 8)
    with pytest.raises(ValueError, match="head_dim"):
        apply_rope(torch.randn(2, 4, 8, 16), cos, sin)
