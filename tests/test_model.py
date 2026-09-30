import torch

from mini_transformer.config import Config
from mini_transformer.model import MiniTransformer

CONFIG = Config()
VOCAB = 128


def _model():
    torch.manual_seed(0)
    return MiniTransformer(vocab_size=VOCAB, config=CONFIG)


def test_forward_shape():
    from mini_transformer.rope import rope_cache

    model = _model()
    cos, sin = rope_cache(CONFIG.block_size, CONFIG.head_dim)
    tokens = torch.randint(0, VOCAB, (2, 16))
    assert model(tokens, cos, sin).shape == (2, 16, VOCAB)


def test_head_is_tied_to_embedding():
    model = _model()
    assert model.lm_head.weight is model.token_embedding.weight


def test_exactly_one_norm_before_head():
    from mini_transformer.rmsnorm import RMSNorm

    model = _model()
    norms = [m for m in model.modules() if isinstance(m, RMSNorm)]
    # n_layer blocks x 2 norms, plus the single final norm
    assert len(norms) == CONFIG.n_layer * 2 + 1


def test_model_is_causal():
    from mini_transformer.rope import rope_cache

    model = _model().eval()
    cos, sin = rope_cache(CONFIG.block_size, CONFIG.head_dim)
    tokens = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        before = model(tokens, cos, sin)
        changed = tokens.clone()
        changed[0, 9:] = torch.randint(0, VOCAB, (7,))
        after = model(changed, cos, sin)
    assert torch.allclose(before[0, :9], after[0, :9], atol=1e-6)
    assert not torch.allclose(before[0, 9:], after[0, 9:], atol=1e-6)


def test_generate_grows_and_crops_to_block_size():
    model = _model().eval()
    out = model.generate(
        torch.randint(0, VOCAB, (1, 4)),
        max_new_tokens=CONFIG.block_size + 10,
        config=CONFIG,
    )
    assert out.shape == (1, 4 + CONFIG.block_size + 10)


def test_transformer_stack_parameter_count_is_vocab_independent():
    """The stack is 1,771,200 params; only the tied embedding scales with vocab."""
    model = _model()
    stack = sum(p.numel() for p in model.blocks.parameters()) + sum(
        p.numel() for p in model.norm.parameters()
    )
    assert stack == 1_771_200


def test_tied_head_means_one_embedding_table():
    model = _model()
    total = sum(p.numel() for p in model.parameters())
    assert total == sum(p.numel() for p in model.token_embedding.parameters()) + 1_771_200
