
from mini_transformer.config import Config
from mini_transformer.model import MiniTransformer
from mini_transformer.summary import main, model_summary


def test_model_summary_calculates_exact_parameters():
    config = Config()
    vocab_size = 128
    model = MiniTransformer(vocab_size=vocab_size, config=config)

    summary = model_summary(model, config, vocab_size)

    assert summary["vocab_size"] == 128
    assert summary["total_params"] == sum(p.numel() for p in model.parameters())
    assert summary["embed_params"] == vocab_size * config.d_model
    assert summary["stack_params"] == 1_771_200
    assert summary["fp32_mb"] > 0
    assert summary["bf16_mb"] == summary["fp32_mb"] / 2
    assert summary["flops_per_token"] == 2 * summary["total_params"]
    assert len(summary["layers"]) == config.n_layer + 3


def test_summary_cli_runs_without_error(capsys):
    main([])
    out = capsys.readouterr().out
    assert "MiniTransformer Architecture Summary" in out
    assert "Total Parameters:" in out
    assert "Transformer Stack:" in out
