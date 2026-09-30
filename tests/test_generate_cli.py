import pytest
import torch

from mini_transformer.config import Config
from mini_transformer.generate import load_checkpoint, main
from mini_transformer.tokenizer import EOT

TEXT = "the quick brown fox jumps over the lazy dog. " * 200


def test_missing_checkpoint_names_the_path(tmp_path):
    with pytest.raises(ValueError, match="smoke.pt"):
        load_checkpoint(tmp_path / "smoke.pt", "cpu", vocab_size=64)


def test_roundtrips_a_checkpoint(tmp_path):
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)

    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "step": 10, "config": config.__dict__}, path)

    loaded, loaded_config = load_checkpoint(path, "cpu", vocab_size=64)
    assert loaded_config == config
    assert loaded.lm_head.weight.shape == (64, config.d_model)


def test_mismatched_config_is_explained(tmp_path):
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "step": 1, "config": config.__dict__}, path)

    other = Config(d_model=256)
    with pytest.raises(ValueError, match="d_model"):
        load_checkpoint(path, "cpu", vocab_size=64, config=other)


def test_the_mismatch_is_named_with_both_values_before_the_model_is_built(tmp_path):
    """`Config(d_model=256)` also trips the attention divisibility check, so a
    diff that runs after `MiniTransformer(...)` reports `d_model 256 is not
    divisible by n_head 6` instead of which hyperparameter actually differs."""
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "step": 1, "config": config.__dict__}, path)

    with pytest.raises(
        ValueError,
        match=r"checkpoint config differs: d_model \(checkpoint 192, requested 256\)",
    ):
        load_checkpoint(path, "cpu", vocab_size=64, config=Config(d_model=256))


def test_weights_that_do_not_fit_the_tokenizer_are_reported(tmp_path):
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "step": 1, "config": config.__dict__}, path)

    with pytest.raises(ValueError, match="do not fit"):
        load_checkpoint(path, "cpu", vocab_size=128)


def _cli(tmp_path):
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer
    from mini_transformer.tokenizer import train_tokenizer

    config = Config(bpe_vocab_size=64, block_size=16)
    tokenizer_path = tmp_path / "tokenizer.json"
    vocab_size = train_tokenizer([TEXT], config.bpe_vocab_size, tokenizer_path)

    model = MiniTransformer(vocab_size=vocab_size, config=config)
    init_weights(model, config.n_layer)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model": model.state_dict(), "step": 3, "config": config.__dict__}, checkpoint
    )

    return [
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(tokenizer_path),
        "--device",
        "cpu",
    ]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_is_seeded_with_eot_instead_of_crashing(tmp_path, capsys, prompt):
    main([*_cli(tmp_path), "--prompt", prompt, "--n", "1"])

    err = capsys.readouterr().err
    assert "blank prompt" in err
    assert EOT in err


def test_prompt_longer_than_the_block_says_it_was_truncated(tmp_path, capsys):
    main([*_cli(tmp_path), "--prompt", TEXT, "--n", "1"])

    assert "only the last 16" in capsys.readouterr().err


def test_the_effective_sampling_settings_are_printed(tmp_path, capsys):
    main(
        [
            *_cli(tmp_path),
            "--prompt",
            "the ",
            "--n",
            "1",
            "--temperature",
            "0.0",
            "--top-p",
            "0.5",
        ]
    )

    err = capsys.readouterr().err
    assert "temperature=0.0" in err
    assert "top_p=0.5" in err


def test_help_documents_the_greedy_sentinel(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])

    assert "greedy" in capsys.readouterr().out


def test_diagnostics_never_pollute_stdout(tmp_path, capsys):
    main([*_cli(tmp_path), "--prompt", "the ", "--n", "1"])

    out = capsys.readouterr().out
    assert "temperature=" not in out
    assert out.startswith("the ")
