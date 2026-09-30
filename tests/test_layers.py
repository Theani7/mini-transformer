import pytest
import torch

from mini_transformer.feed_forward import SwiGLU
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


def test_swiglu_shape_and_gating():
    ffn = SwiGLU(16, 32)
    x = torch.randn(2, 3, 16)
    assert ffn(x).shape == (2, 3, 16)
    with torch.no_grad():
        ffn.up.weight.zero_()
    # gate is multiplied by up, so zeroing up must zero the output
    assert torch.allclose(ffn(x), torch.zeros(2, 3, 16))
