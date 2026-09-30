import json
from dataclasses import FrozenInstanceError

import pytest

from mini_transformer.config import Config


def test_defaults_match_measured_configuration():
    c = Config()
    assert (c.d_model, c.n_layer, c.n_head, c.d_ff) == (192, 4, 6, 512)
    assert (c.block_size, c.batch_size, c.iters) == (128, 8, 6000)
    assert c.head_dim == 32
    assert c.min_lr == pytest.approx(6e-4)


def test_roundtrips_through_disk(tmp_path):
    path = tmp_path / "config.json"
    Config().save(path)
    assert json.loads(path.read_text())["d_model"] == 192
    assert Config.load(path) == Config()


def test_is_frozen():
    with pytest.raises(FrozenInstanceError):
        Config().d_model = 1


from torch import nn

from mini_transformer.init import init_weights


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.proj = nn.Linear(64, 64, bias=False)
        self.feed_forward = nn.Module()
        self.feed_forward.down = nn.Linear(64, 64, bias=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(100, 64)
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])
        self.head = nn.Linear(64, 100, bias=False)


def test_embedding_and_linear_use_std_002():
    m = init_weights(_Model(), n_layer=4)
    assert m.tok.weight.std().item() == pytest.approx(0.02, abs=0.005)
    assert m.head.weight.std().item() == pytest.approx(0.02, abs=0.005)


def test_residual_projections_are_scaled_down_by_depth():
    m = init_weights(_Model(), n_layer=4)
    expected = 0.02 / (2 * 4) ** 0.5
    assert m.blocks[0].attention.proj.weight.std().item() == pytest.approx(expected, abs=0.002)
    assert m.blocks[0].feed_forward.down.weight.std().item() == pytest.approx(expected, abs=0.002)
