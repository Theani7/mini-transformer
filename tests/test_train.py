import json
from dataclasses import asdict, dataclass

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


def test_resume_never_downgrades_a_finished_checkpoint(monkeypatch, tmp_path, capsys):
    out = tmp_path / "m.pt"
    built = _spy(monkeypatch, "build_optimizer")

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])
    assert torch.load(out, weights_only=True)["step"] == 5

    capsys.readouterr()
    built.clear()
    _run(monkeypatch, tmp_path, ["--iters", "3", "--device", "cpu", "--resume"])

    assert torch.load(out, weights_only=True)["step"] == 5
    assert built == []
    assert "nothing to do" in capsys.readouterr().out


def test_stale_cache_is_retrained_when_the_corpus_changes(monkeypatch, tmp_path):
    packs = _spy(monkeypatch, "pack_ids")

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu", "--corpus-chars", "400"])
    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu", "--corpus-chars", "9000"])

    assert len(packs) == 2
    assert Config.load(tmp_path / "data" / "config.json").corpus_chars == 9000


def test_a_stamp_from_another_version_is_treated_as_a_cache_miss(monkeypatch, tmp_path):
    """`Config` lost a field in this task, which made every `data/config.json`
    written by an earlier version unloadable. A stamp is a cache: the fix is to
    rebuild it, not to crash the run before training starts."""
    packs = _spy(monkeypatch, "pack_ids")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stale = asdict(Config())
    stale["eval_interval"] = 500
    (data_dir / "config.json").write_text(json.dumps(stale))

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu"])

    assert len(packs) == 1
    rebuilt = json.loads((data_dir / "config.json").read_text())
    assert "eval_interval" not in rebuilt
    assert rebuilt["corpus_chars"] == Config().corpus_chars


def test_iteration_flags_do_not_invalidate_the_cache(monkeypatch, tmp_path):
    packs = _spy(monkeypatch, "pack_ids")

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])
    _run(monkeypatch, tmp_path, ["--iters", "9", "--device", "cpu", "--batch-size", "2"])

    assert len(packs) == 1


def test_the_head_is_sized_from_the_tokenizer_not_the_requested_vocab(monkeypatch, tmp_path):
    """`tokenizer.get_vocab_size()` is what the head is built from, because the
    corpus only yielded 6,582 of the 8,192 requested merges. Sizing from
    `config.bpe_vocab_size` instead trains a head with 1,610 rows the tokenizer
    can never emit, and the last logit is dead.

    Nothing else pinned this: Task 2's test guards the tokenizer's *return
    value*, Task 9's guards the generate CLI, and the trainer's *use* of it had
    no guard at all - the wrong value passed the whole suite. The `!=` below is
    what keeps that from becoming a vacuous assertion if the test corpus ever
    trains to exactly `bpe_vocab_size` merges.
    """
    from mini_transformer.model import MiniTransformer
    from mini_transformer.tokenizer import load_tokenizer

    seen = []
    real_init = MiniTransformer.__init__

    def spy(self, vocab_size, config):
        seen.append(vocab_size)
        real_init(self, vocab_size, config)

    monkeypatch.setattr(MiniTransformer, "__init__", spy)

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu"])

    requested = _FastConfig().bpe_vocab_size
    assert seen == [load_tokenizer(tmp_path / "data" / "tokenizer.json").get_vocab_size()]
    assert seen[0] != requested


def test_resume_rejects_a_checkpoint_trained_with_different_flags(monkeypatch, tmp_path):
    """`--resume` used to read only `state["model"]` and `state["step"]`, so this
    silently resumed under the NEW hyperparameters and then overwrote the saved
    `config` - the mismatch `generate.load_checkpoint` exists to refuse."""
    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu", "--batch-size", "8"])

    with pytest.raises(ValueError, match=r"batch_size \(checkpoint 8, requested 2\)"):
        _run(
            monkeypatch,
            tmp_path,
            ["--iters", "9", "--device", "cpu", "--batch-size", "2", "--resume"],
        )


def test_resume_can_still_extend_a_run_to_a_higher_iters(monkeypatch, tmp_path):
    """The counterweight to the strict diff: `iters` is exempted, because raising
    it is the entire purpose of `--resume`. Without the exemption the diff would
    reject its own reason for existing."""
    _run(monkeypatch, tmp_path, ["--iters", "3", "--device", "cpu"])

    _run(monkeypatch, tmp_path, ["--iters", "7", "--device", "cpu", "--resume"])

    assert torch.load(tmp_path / "m.pt", weights_only=True)["step"] == 7


def test_resume_reports_weights_that_do_not_fit_as_a_value_error(monkeypatch, tmp_path):
    """The identical call in `generate.py` is wrapped in a `ValueError`, which
    both CLIs turn into a one-line message. Bare here, `--resume` on a
    mismatched checkpoint dumps a raw `RuntimeError` traceback."""
    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu"])

    out = tmp_path / "m.pt"
    state = torch.load(out, weights_only=True)
    # A wrong-shaped tensor that no `Config` field controls, so the config
    # check passes and the failure can only come from the weight load.
    state["model"]["token_embedding.weight"] = torch.zeros(4, 192)
    torch.save(state, out)

    with pytest.raises(ValueError, match="checkpoint weights do not fit"):
        _run(monkeypatch, tmp_path, ["--iters", "9", "--device", "cpu", "--resume"])


def test_a_truncated_stamp_is_treated_as_a_cache_miss(monkeypatch, tmp_path):
    """`Config.save` is a non-atomic `write_text`, so a crash mid-write leaves
    truncated JSON behind, and `Config.load` raises `JSONDecodeError` - a
    `ValueError`, which the guard above did not catch. The comment on that guard
    claimed a format change should invalidate the cache rather than crash it."""
    packs = _spy(monkeypatch, "pack_ids")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text('{"d_model": 192, "n_lay')

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu"])

    assert len(packs) == 1
    assert Config.load(data_dir / "config.json").corpus_chars == Config().corpus_chars
