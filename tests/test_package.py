import subprocess
import sys
from pathlib import Path

import pytest

import mini_transformer

# Derived from the filesystem, not hand-maintained: a hardcoded list is the same
# hole one level up, since adding a module would leave it silently uncovered.
# `rglob` reaches subpackages too, so names are built from the path relative to
# the package root - `sub/bar.py` has to import as `mini_transformer.sub.bar`,
# not `mini_transformer.bar`.
PACKAGE_DIR = Path(mini_transformer.__file__).parent

MODULES = sorted(
    ".".join(path.relative_to(PACKAGE_DIR).with_suffix("").parts)
    for path in PACKAGE_DIR.rglob("*.py")
    if path.stem != "__init__"
)

IMPORT_ALL = ", ".join(f"mini_transformer.{name}" for name in MODULES)

IMPORT_TIMEOUT = 30


def test_every_module_imports():
    # Subsumed by the subprocess test below, which imports the same set and
    # asserts returncode == 0. Kept because an ImportError names the offending
    # module here, where the subprocess only reports it inside stderr.
    assert MODULES, "found no modules to import - has the package moved?"
    for name in MODULES:
        __import__(f"mini_transformer.{name}")


def test_importing_every_module_prints_nothing():
    """Importing must be silent, and must not do measurable work.

    `multi_head_attention` once built tensors and printed at module scope, so
    `import mini_transformer.train` ran 1000 optimizer steps as a side effect of
    being imported. A subprocess is the only way to see stdout from import time:
    pytest has already imported the modules in-process by this point.

    Two separate checks, and the second is much weaker than it looks. Silence is
    enforced exactly. Import-time *work* is only caught above
    `IMPORT_TIMEOUT` - cheap silent work (a few seconds of tensor math, printing
    nothing) still passes. A cold import is ~0.9s, so 30s is ~30x headroom and
    will not flake, but it means a real training loop at module scope is caught
    while a toy one is not. Don't read the timeout as a general claim.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {IMPORT_ALL}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=IMPORT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"importing the package took over {IMPORT_TIMEOUT}s - "
            f"something is running at module scope"
        )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "", result.stdout


def test_package_exports_public_api():
    """`mini_transformer` must export primary classes and helpers at the package root."""
    import mini_transformer

    assert hasattr(mini_transformer, "MiniTransformer")
    assert hasattr(mini_transformer, "Config")
    assert hasattr(mini_transformer, "encode")
    assert hasattr(mini_transformer, "decode")
    assert hasattr(mini_transformer, "load_tokenizer")
    assert hasattr(mini_transformer, "__version__")
    assert mini_transformer.__version__ == "0.2.0"


def test_console_scripts_resolve_to_callables():
    """pyproject.toml script entries must point to importable, callable functions.

    Prevents typos in console script endpoints that would break `uv run <script>`.
    """
    import importlib
    import tomllib

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    scripts = data["project"]["scripts"]
    assert "mini-summary" in scripts
    for name, target in scripts.items():
        mod_name, func_name = target.split(":")
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, func_name), f"{name} points to missing attribute {target}"
        assert callable(getattr(mod, func_name)), f"{target} is not callable"


