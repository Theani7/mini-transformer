from dataclasses import dataclass

import pytest
import torch

from mini_transformer.config import Config
from mini_transformer.init import init_weights
from mini_transformer.train import build_optimizer, lr_at, resolve_device

TEXT = "the quick brown fox jumps over the lazy dog. " * 200


@dataclass(frozen=True)
class _FastConfig(Config):
    block_size: int = 8
    sample_interval: int = 2
    warmup: int = 2


def _run(monkeypatch, tmp_path, argv):
    from mini_transformer import train

    monkeypatch.setattr(train, "Config", _FastConfig)
    monkeypatch.setattr(train, "load_corpus", lambda d, c, chars: TEXT[:chars])
    train.main(
        [
            *argv,
            "--data-dir",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "m.pt"),
        ]
    )


def _spy(monkeypatch, name):
    from mini_transformer import train

    calls = []
    real = getattr(train, name)
    monkeypatch.setattr(
        train, name, lambda *a, **k: (calls.append(a), real(*a, **k))[1]
    )
    return calls


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
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {
        id(p) for p in model.parameters()
    }


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


def test_main_samples_on_every_sample_interval_boundary(monkeypatch, tmp_path):
    calls = _spy(monkeypatch, "_sample")

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])

    assert [a[4] for a in calls] == ["", ""]


def test_main_applies_the_schedule_to_every_param_group(monkeypatch, tmp_path):
    from mini_transformer import train

    seen = []
    real_build = train.build_optimizer

    def build(model, config):
        opt = real_build(model, config)
        real_step = opt.step

        def step(*a, **k):
            seen.append([g["lr"] for g in opt.param_groups])
            return real_step(*a, **k)

        opt.step = step
        return opt

    monkeypatch.setattr(train, "build_optimizer", build)

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])

    c = _FastConfig(iters=5)
    assert seen == [[pytest.approx(lr_at(c, s))] * 2 for s in range(5)]


def test_resume_never_downgrades_a_finished_checkpoint(monkeypatch, tmp_path):
    out = tmp_path / "m.pt"

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])
    assert torch.load(out, weights_only=True)["step"] == 5

    _run(monkeypatch, tmp_path, ["--iters", "3", "--device", "cpu", "--resume"])
    assert torch.load(out, weights_only=True)["step"] == 5


def test_stale_cache_is_retrained_when_the_corpus_changes(monkeypatch, tmp_path):
    packs = _spy(monkeypatch, "pack_ids")

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu", "--corpus-chars", "400"])
    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu", "--corpus-chars", "9000"])

    assert len(packs) == 2
    assert Config.load(tmp_path / "data" / "config.json").corpus_chars == 9000


def test_iteration_flags_do_not_invalidate_the_cache(monkeypatch, tmp_path):
    packs = _spy(monkeypatch, "pack_ids")

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])
    _run(monkeypatch, tmp_path, ["--iters", "9", "--device", "cpu", "--batch-size", "2"])

    assert len(packs) == 1
