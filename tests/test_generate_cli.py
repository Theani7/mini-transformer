import sys

import pytest
import torch

from mini_transformer.config import Config
from mini_transformer.generate import cli, load_checkpoint, main
from mini_transformer.tokenizer import EOT, encode, load_tokenizer

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


def test_roundtrips_a_safetensors_checkpoint(tmp_path):
    import json

    from safetensors.torch import save_model

    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)

    path = tmp_path / "model.safetensors"
    save_model(model, path, metadata={"config": json.dumps(config.__dict__), "step": "10"})

    loaded, loaded_config = load_checkpoint(path, "cpu", vocab_size=64)
    assert loaded_config == config
    assert loaded.lm_head.weight.shape == (64, config.d_model)
    assert loaded.lm_head.weight is loaded.token_embedding.weight


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

    with pytest.raises(ValueError) as excinfo:
        load_checkpoint(path, "cpu", vocab_size=128)

    message = str(excinfo.value)
    assert "trained on a 64-token vocabulary" in message
    assert "128 tokens were requested" in message
    assert "--tokenizer" in message
    assert "requested config" not in message


def test_a_checkpoint_from_another_version_names_the_unknown_field(tmp_path):
    """A checkpoint written before a `Config` field was removed used to raise a
    bare `TypeError`, which `__main__` does not catch - so the user got a
    traceback instead of the project's clean one-line error, and `rm -rf data`
    could not help because the checkpoint is the deliverable."""
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 1,
            "config": {**config.__dict__, "retired_field": 1},
        },
        path,
    )

    with pytest.raises(ValueError) as excinfo:
        load_checkpoint(path, "cpu", vocab_size=64)

    message = str(excinfo.value)
    assert "retired_field" in message
    assert "written by a different version" in message
    assert "retrain" in message


def test_a_checkpoint_from_another_version_names_the_unknown_field_even_when_a_config_is_passed(tmp_path):
    """The caller's config skips the `Config(**saved)` line entirely, so the
    guard has to be in front of the diff loop too - otherwise the unknown key
    surfaces as `AttributeError: 'Config' object has no attribute ...`."""
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    config = Config()
    model = MiniTransformer(vocab_size=64, config=config)
    init_weights(model, config.n_layer)
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 1,
            "config": {**config.__dict__, "retired_field": 1},
        },
        path,
    )

    with pytest.raises(ValueError, match="retired_field"):
        load_checkpoint(path, "cpu", vocab_size=64, config=Config())


def test_a_checkpoint_with_no_config_key_loads_but_says_so(tmp_path, capsys):
    """An older or hand-written checkpoint may carry weights and step but no
    `config`. It still loads - but under the *current* defaults, which may not be
    what it was trained with, so the design spec asks for a warning here."""
    from mini_transformer.init import init_weights
    from mini_transformer.model import MiniTransformer

    model = MiniTransformer(vocab_size=64, config=Config())
    init_weights(model, Config().n_layer)
    path = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "step": 1}, path)

    loaded, loaded_config = load_checkpoint(path, "cpu", vocab_size=64)

    assert loaded_config == Config()
    assert loaded.lm_head.weight.shape == (64, Config().d_model)
    assert "stored no config" in capsys.readouterr().err


@pytest.mark.parametrize(
    "state",
    [
        pytest.param({"step": 1}, id="no-model-key"),
        pytest.param({"model": {"weight": torch.zeros(4, 4)}, "step": 1}, id="no-embedding"),
    ],
)
def test_a_foreign_state_dict_is_a_value_error_not_a_bare_key_error(tmp_path, state):
    """`cli()`'s own docstring promises `main` raises `ValueError` for anything
    the user can fix, and `mini-transformer` is the first command a user runs.
    A foreign or legacy `state_dict` used to escape as a bare `KeyError`, which
    `cli` does not catch, so the console script printed a traceback."""
    path = tmp_path / "model.pt"
    torch.save(state, path)

    with pytest.raises(ValueError, match="is not a mini-transformer checkpoint"):
        load_checkpoint(path, "cpu", vocab_size=64)


def test_a_foreign_state_dict_reaches_the_console_script_as_a_one_line_message(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.pt"
    torch.save({"step": 1}, checkpoint)
    tokenizer_path = tmp_path / "tokenizer.json"
    from mini_transformer.tokenizer import train_tokenizer

    train_tokenizer([TEXT], 64, tokenizer_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mini-transformer",
            "--checkpoint",
            str(checkpoint),
            "--tokenizer",
            str(tokenizer_path),
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        cli()

    assert "is not a mini-transformer checkpoint" in str(exit_info.value)


def test_a_missing_tokenizer_names_the_path_instead_of_a_traceback(tmp_path):
    with pytest.raises(ValueError, match="tokenizer.json not found - run:"):
        main(
            [
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--tokenizer",
                str(tmp_path / "tokenizer.json"),
                "--device",
                "cpu",
            ]
        )


def test_the_console_script_reports_a_missing_tokenizer_without_a_traceback(
    tmp_path, monkeypatch
):
    """The `mini-transformer` console script calls `cli`, not `main`, so the
    `ValueError` -> one-line message conversion has to happen somewhere both can reach.
    It lives in `cli`; `__main__` routes through `cli` too. Without it the console script
    dumps a traceback, since a console script never executes the package's `__main__`."""
    monkeypatch.setattr(
        sys, "argv", ["mini-transformer", "--tokenizer", str(tmp_path / "nope.json")]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli()
    assert "nope.json not found - run:" in str(exit_info.value)


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


def _prompt_of_token_length(tmp_path, n):
    tokenizer = load_tokenizer(tmp_path / "tokenizer.json")
    built = ""
    for char in TEXT:
        built += char
        if len(encode(tokenizer, built)) == n:
            return built
    raise AssertionError(f"no prefix of TEXT encodes to exactly {n} tokens")


def _spy_generate(monkeypatch):
    from mini_transformer.model import MiniTransformer

    seen = {}

    def spy(self, tokens, **kwargs):
        seen["tokens"] = tokens
        seen.update(kwargs)
        return tokens

    monkeypatch.setattr(MiniTransformer, "generate", spy)
    return seen


@pytest.mark.parametrize("prompt", ["", "   "])
def test_blank_prompt_is_seeded_with_eot_instead_of_crashing(tmp_path, capsys, prompt):
    main([*_cli(tmp_path), "--prompt", prompt, "--n", "1"])

    err = capsys.readouterr().err
    assert "blank prompt" in err
    assert EOT in err


@pytest.mark.parametrize("tokens", [16, 17])
def test_the_truncation_warning_fires_only_past_the_block(tmp_path, capsys, tokens):
    main(
        [*_cli(tmp_path), "--prompt", _prompt_of_token_length(tmp_path, tokens), "--n", "1"]
    )

    warned = "only the last 16" in capsys.readouterr().err
    assert warned is (tokens > 16)


def test_the_flags_and_the_seed_reach_generate(tmp_path, monkeypatch):
    seen = _spy_generate(monkeypatch)

    main(
        [
            *_cli(tmp_path),
            "--prompt",
            "",
            "--n",
            "7",
            "--temperature",
            "0.3",
            "--top-p",
            "0.5",
        ]
    )

    assert seen["temperature"] == 0.3
    assert seen["top_p"] == 0.5
    assert seen["max_new_tokens"] == 7
    assert seen["tokens"][0].tolist() == [
        load_tokenizer(tmp_path / "tokenizer.json").token_to_id(EOT)
    ]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="CPU is the torch default device"
)
def test_the_prompt_tensor_is_built_on_the_resolved_device(tmp_path, monkeypatch):
    seen = _spy_generate(monkeypatch)

    main([*_cli(tmp_path), "--device", "mps", "--prompt", "the ", "--n", "1"])

    assert seen["tokens"].device.type == "mps"


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


def test_use_cache_flag_reaches_generate(tmp_path, monkeypatch):
    seen = _spy_generate(monkeypatch)

    main([*_cli(tmp_path), "--prompt", "the ", "--n", "1", "--use-cache"])

    assert seen["use_cache"] is True


def test_stream_flag_prints_output(tmp_path, capsys):
    main([*_cli(tmp_path), "--prompt", "the ", "--n", "2", "--stream"])

    out = capsys.readouterr().out
    assert out.startswith("the ")


def test_top_k_and_repetition_penalty_flags_reach_generate(tmp_path, monkeypatch):
    seen = _spy_generate(monkeypatch)

    main(
        [
            *_cli(tmp_path),
            "--prompt",
            "the ",
            "--n",
            "1",
            "--top-k",
            "5",
            "--repetition-penalty",
            "1.2",
        ]
    )

    assert seen["top_k"] == 5
    assert seen["repetition_penalty"] == 1.2


def test_interactive_mode_runs_prompts_and_exits(tmp_path, monkeypatch, capsys):
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    main([*_cli(tmp_path), "-i", "--n", "1"])

    out = capsys.readouterr().out
    assert "interactive mode" in out


def test_stop_on_eot_flag_passes_eos_token_id_to_generate(tmp_path, monkeypatch):
    seen = _spy_generate(monkeypatch)
    cli_args = _cli(tmp_path)

    main([*cli_args, "--prompt", "the ", "--n", "1", "--stop-on-eot"])

    eot_id = load_tokenizer(tmp_path / "tokenizer.json").token_to_id(EOT)
    assert seen["eos_token_id"] == {eot_id}


def test_chat_mode_runs_conversation_turn(tmp_path, monkeypatch, capsys):
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    main([*_cli(tmp_path), "--chat", "--n", "1"])

    out = capsys.readouterr().out
    assert "ChatML mode" in out
    assert "Assistant: " in out


def test_clean_text_normalizes_formatting():
    from mini_transformer.generate import clean_text

    raw = "The espai no Hana 2 @,@ 000 was a ( profound ) walls , and it ' s good ."
    cleaned = clean_text(raw)
    assert "@,@" not in cleaned
    assert "2, 000" in cleaned
    assert "(profound)" in cleaned
    assert "walls, and" in cleaned
    assert "it's" in cleaned


def test_clean_flag_normalizes_output(tmp_path, capsys):
    main([*_cli(tmp_path), "--prompt", "the ", "--n", "2", "--clean"])

    out = capsys.readouterr().out
    assert out.startswith("the ")


