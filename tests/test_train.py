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
    """A `data/config.json` written by a different `Config` shape is a cache
    miss, not a crash. Renaming or removing a `Config` field makes every
    stamp written before it unloadable, and the fix is to rebuild the cache
    rather than fail the run before training starts."""
    packs = _spy(monkeypatch, "pack_ids")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    stale = asdict(Config())
    stale["retired_field"] = 1
    (data_dir / "config.json").write_text(json.dumps(stale))

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu"])

    assert len(packs) == 1
    rebuilt = json.loads((data_dir / "config.json").read_text())
    assert "retired_field" not in rebuilt
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


def test_resume_restores_optimizer_state(monkeypatch, tmp_path):
    """Regression: Checkpoints previously omitted `optimizer`, causing `--resume` to reset
    AdamW momentum and variance moments to zero mid-training."""
    _run(monkeypatch, tmp_path, ["--iters", "3", "--device", "cpu"])

    state = torch.load(tmp_path / "m.pt", weights_only=True)
    assert "optimizer" in state
    assert len(state["optimizer"]["state"]) > 0

    _run(monkeypatch, tmp_path, ["--iters", "5", "--device", "cpu", "--resume"])
    state_after = torch.load(tmp_path / "m.pt", weights_only=True)
    assert state_after["step"] == 5
    assert "optimizer" in state_after


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


def test_validation_is_opt_in_so_the_default_run_uses_the_whole_corpus(monkeypatch, tmp_path):
    """Default eval_interval is 0. Holding out 5% of the corpus for a number
    nothing reads would be a silent 5% tax on the deliverable."""
    packs = _spy(monkeypatch, "pack_ids")

    _run(monkeypatch, tmp_path, ["--iters", "1", "--device", "cpu"])

    assert len(packs) == 1
    assert not (tmp_path / "data" / "val.bin").exists()
    assert not (tmp_path / "best.pt").exists()


def test_asking_for_validation_holds_out_a_val_set_and_writes_best(monkeypatch, tmp_path):
    packs = _spy(monkeypatch, "pack_ids")

    _run(
        monkeypatch,
        tmp_path,
        ["--iters", "1", "--device", "cpu", "--eval-interval", "1", "--best-out", str(tmp_path / "best.pt")],
    )

    assert len(packs) == 2
    assert (tmp_path / "data" / "val.bin").exists()
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert "val_loss" in best


def test_toggling_validation_rebuilds_the_split_but_retuning_it_does_not(monkeypatch, tmp_path):
    """The split changes train.bin, so turning eval on must repack. The eval
    cadence does not touch the bytes, so changing only that must not retokenize."""
    packs = _spy(monkeypatch, "pack_ids")
    base = ["--iters", "1", "--device", "cpu"]

    _run(monkeypatch, tmp_path, [*base, "--eval-interval", "5"])
    assert len(packs) == 2  # train + val

    _run(monkeypatch, tmp_path, [*base, "--eval-interval", "9"])
    assert len(packs) == 2  # cadence only, cache still valid

    _run(monkeypatch, tmp_path, [*base, "--eval-interval", "0"])
    assert len(packs) == 3  # dropping the split changes train.bin


def test_a_resume_does_not_overwrite_a_better_best_checkpoint(monkeypatch, tmp_path):
    """Regression: `best_val` was reset to None on resume, so the first eval of
    the second run overwrote `best.pt` unconditionally - even when the resumed
    model was worse, which is exactly when the old best is worth keeping.

    Driving this from loss dynamics is flaky, so the previous run's best is
    set to an implausibly good score directly. Any real eval must lose to it.
    """
    best_out = str(tmp_path / "best.pt")
    first = ["--device", "cpu", "--eval-interval", "2", "--best-out", best_out]

    _run(monkeypatch, tmp_path, [*first, "--iters", "8"])
    before = torch.load(best_out, weights_only=True)

    # Stand in for a previous run whose best was excellent. Zero is unbeatable,
    # so the guard is exercised without depending on loss dynamics.
    previous = torch.load(tmp_path / "m.pt", weights_only=True)
    previous["best_val"] = 0.0
    previous["best_step"] = before["step"]
    torch.save(previous, tmp_path / "m.pt")

    _run(monkeypatch, tmp_path, [*first, "--iters", "24", "--resume"])

    # The resumed run can never beat 0.0, so best.pt must come through intact
    # rather than being replaced by whatever the first post-resume eval found.
    after = torch.load(best_out, weights_only=True)
    assert after["val_loss"] == pytest.approx(before["val_loss"])
    assert after["step"] == before["step"]

    # and the knowledge travels in the final checkpoint, not just in memory
    resumed = torch.load(tmp_path / "m.pt", weights_only=True)
    assert resumed["best_val"] == pytest.approx(0.0)
    assert resumed["best_step"] == before["step"]


def test_turning_validation_on_at_resume_warns_the_number_is_contaminated(monkeypatch, tmp_path, capsys):
    """The val tail was in the training data up to the resume point, so its
    loss is not a held-out score and saying so beats a quietly wrong number."""
    _run(monkeypatch, tmp_path, ["--iters", "4", "--device", "cpu"])

    _run(
        monkeypatch,
        tmp_path,
        ["--iters", "8", "--device", "cpu", "--eval-interval", "2", "--resume"],
    )

    assert "contaminated" in capsys.readouterr().out


def test_resolve_device_picks_cuda_when_available(monkeypatch):
    from mini_transformer.train import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_early_stopping_halts_training_when_patience_exceeded(monkeypatch, tmp_path, capsys):
    from mini_transformer import train

    # Return increasing validation loss so patience is exhausted
    eval_call_count = 0

    def mock_eval(*args, **kwargs):
        nonlocal eval_call_count
        eval_call_count += 1
        return float(eval_call_count)

    monkeypatch.setattr(train, "evaluate", mock_eval)

    _run(
        monkeypatch,
        tmp_path,
        [
            "--iters", "20",
            "--device", "cpu",
            "--eval-interval", "2",
            "--early-stopping-patience", "2",
        ],
    )

    out = capsys.readouterr().out
    assert "early stopping" in out
    state = torch.load(tmp_path / "m.pt", weights_only=True)
    assert state["step"] < 20


def test_train_writes_safetensors_checkpoint(monkeypatch, tmp_path):
    """Training emits both .pt and .safetensors checkpoints alongside each other."""
    from safetensors import safe_open

    _run(monkeypatch, tmp_path, ["--iters", "2", "--device", "cpu"])

    safetensors_path = tmp_path / "m.safetensors"
    assert safetensors_path.exists()
    with safe_open(safetensors_path, framework="pt") as f:
        meta = f.metadata() or {}
        assert "config" in meta
        assert "step" in meta


def test_mixed_precision_flag_trains_successfully(monkeypatch, tmp_path, capsys):
    _run(
        monkeypatch,
        tmp_path,
        ["--iters", "2", "--device", "cpu", "--mixed-precision", "bf16"],
    )

    out = capsys.readouterr().out
    assert "mixed_precision=bf16" in out
    assert (tmp_path / "m.pt").exists()

