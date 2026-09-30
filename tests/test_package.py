import importlib
import subprocess
import sys

MODULES = [
    "config",
    "data",
    "feed_forward",
    "generate",
    "init",
    "model",
    "multi_head_attention",
    "rmsnorm",
    "rope",
    "sampling",
    "tokenizer",
    "train",
    "transformer_block",
]

IMPORT_ALL = ", ".join(f"mini_transformer.{name}" for name in MODULES)


def test_every_module_imports():
    for name in MODULES:
        importlib.import_module(f"mini_transformer.{name}")


def test_importing_every_module_prints_nothing():
    """Importing must be silent.

    `multi_head_attention` once built tensors and printed at module scope, so
    `import mini_transformer.train` ran 1000 optimizer steps as a side effect of
    being imported. A subprocess is the only way to see stdout from import time:
    pytest has already imported the modules in-process by this point.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {IMPORT_ALL}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "", result.stdout
