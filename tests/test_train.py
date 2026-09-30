import pytest
import torch

from mini_transformer.config import Config
from mini_transformer.init import init_weights
from mini_transformer.train import build_optimizer, lr_at, resolve_device


def test_lr_warms_up_then_decays_to_minimum():
    c = Config()
    assert lr_at(c, 0) == pytest.approx(0.0)
    assert lr_at(c, c.warmup) == pytest.approx(c.lr)
    assert lr_at(c, c.iters) == pytest.approx(c.min_lr, rel=0.02)
    mid = lr_at(c, c.warmup + (c.iters - c.warmup) // 2)
    assert c.min_lr < mid < c.lr


def test_decay_groups_split_by_dimension():
    c = Config()
    from mini_transformer.model import MiniTransformer

    model = MiniTransformer(vocab_size=64, config=c)
    opt = build_optimizer(model, c)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == c.weight_decay
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in opt.param_groups[0]["params"])
    assert all(p.dim() < 2 for p in opt.param_groups[1]["params"])


def test_resolve_device_honours_explicit_request():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_never_returns_an_unknown_device():
    assert resolve_device("auto") in {"cpu", "mps"}


def test_init_weights_gives_the_tied_head_std_002():
    from mini_transformer.model import MiniTransformer

    c = Config()
    torch.manual_seed(0)
    model = init_weights(MiniTransformer(vocab_size=64, config=c), c.n_layer)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.token_embedding.weight.std().item() == pytest.approx(0.02, abs=0.005)