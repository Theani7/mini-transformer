import subprocess
import sys

import torch

from mini_transformer.model import MiniTransformer


def test_forward_produces_logits():
    model = MiniTransformer(8, 32, 4, 128, 2, 64)
    logits = model(torch.tensor([[3, 2, 4, 4, 5]]))
    assert logits.shape == (1, 5, 8)


def test_generate_grows_sequence_and_respects_max_seq_len():
    model = MiniTransformer(8, 16, 2, 32, 1, max_seq_len=4)
    out = model.generate(torch.tensor([[1, 2]]), max_new_tokens=10)
    assert out.shape == (1, 12)


def test_generate_samples_at_nonzero_temperature():
    torch.manual_seed(0)
    model = MiniTransformer(8, 16, 2, 32, 1, max_seq_len=16)
    out = model.generate(
        torch.tensor([[1]]),
        max_new_tokens=5,
        temperature=0.8
    )
    assert out.shape == (1, 6)
    assert 0 <= out.min() and out.max() < 8


def test_importing_package_is_silent():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mini_transformer.model, mini_transformer.transformer_block",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", result.stdout
