# AGENTS.md

## Commands

```bash
uv sync                                  # dev tools (pytest, ruff) are a dependency-group, not main deps
uv run ruff check .                      # lint first — CI runs lint, then test
uv run pytest -q                         # 129 tests, ~12s, fully offline
uv run pytest tests/test_eval.py -q      # one file
uv run pytest -q -k "resume"             # one behaviour
uv build                                 # wheel + sdist
```

- The suite needs **no network and no HF cache** (`load_dataset` is stubbed in tests). Only a
  real `train` run downloads the corpus.
- CI runs on `ubuntu-latest` only and uses `uv sync --frozen`. **Adding a dependency without
  refreshing `uv.lock` breaks CI**, not just the local env.
- Ruff's `extend-select = ["E402", "B"]` is deliberate (see the comment in `pyproject.toml`) —
  default `ruff check .` passes on code this repo rejects.

## Layout

- **Flat package.** `[tool.uv.build-backend] module-root = ""` means a new top-level directory
  changes what lands in the wheel. Keep new code inside `mini_transformer/`.
- `expirements/` (sic) is **teaching scripts, not source** — do not refactor them into the
  package. `01_langauge_model.py` and `02_self_attention.py` are self-contained torch-only;
  `03_rope.py`, `04_rmsnorm.py`, `05_swiglu.py` import from `mini_transformer` and need it installed.
- `my_env/` is a stray untracked venv that ignores itself. `docs/superpowers/` is gitignored
  superpowers tooling, not project documentation. Neither is part of the project.
- `data/` and `*.pt` are gitignored, so deleting `data/` is the intended fix for a bad
  tokenizer or packed corpus.

## Invariants that are easy to break

- **`Config` (`config.py`) is the single source of truth** — a frozen dataclass round-tripped
  through JSON. Adding a field makes every existing checkpoint *and* the data stamp
  un-loadable by design.
- **`resolve_checkpoint_config` is the shared validator for both checkpoint loaders**
  (`train` resume and `generate`). Both must keep routing through it; a second hand-rolled
  check recreates the "two loaders, two contracts" split it exists to prevent.
- **`data/config.json` is a cache stamp, not config.** `train.py` loads it inside
  `try/except (TypeError, ValueError)` so a format change rebuilds instead of crashing. If a
  new `Config` field changes the *packed bytes*, add it to `CACHE_KEY` in `train.py` or stale
  data is silently reused. The key describes what was written, not how the run was scheduled
  — `eval_interval` is deliberately excluded so retuning it does not force a retokenize.
- **Always call `init_weights(model, config.n_layer)` after constructing `MiniTransformer`.**
  Default `nn.Embedding` init yields loss ≈175 instead of ≈4.3.
- **`generate.py` runs `cli()` under `__main__`, not `main()`** — that is what converts
  `ValueError` into a one-line message. `train.py` calls `main()` and does show tracebacks.
  The asymmetry is real; do not assume the two CLIs behave the same.
- **`generate.py` imports `resolve_device` from `train.py`**, so the dependency runs
  generate → train. Never make `train` import `generate`.
- The LM head is **tied** to the token embedding (`lm_head.weight = token_embedding.weight`)
  and `init_weights` preserves the tie. Do not split it into two parameters.
- Packed ids are `uint16`, capping the vocabulary at 65,535 (guarded in `tokenizer.py`).
  `load_packed` requires `block_size + 2` tokens, and `split_ids` guarantees both halves meet
  that; too small a corpus raises rather than silently producing a bad batch.

## Testing conventions

- **Name tests after the regression they prevent** and give them a docstring citing the bug.
  That is the house style here, not optional polish.
- Anything touching the training loop should drive the real `train.main()` through the
  `test_train.py` idiom: monkeypatch `train.Config` to a `_FastConfig` (tiny `block_size` /
  `iters`) and `train.load_corpus` to a fixed string. That keeps it offline and ~1s.
- The one MPS test is `skipif(not torch.backends.mps.is_available())`, so it **silently does not
  run on Linux CI**. Verify MPS paths locally on macOS.
- Tests that stand in for "written by a different version" use a literal `retired_field` key.
  Do not reuse a real `Config` field name for that — it stops being an unknown-key case the
  moment that field exists.

## Do not "fix" these

- **Rising validation loss is the expected result, not a bug.** This is a memorisation demo:
  train loss falls while val loss climbs once the model starts memorising. `checkpoints/model.pt`
  (final) is the checkpoint that writes readable text; `best.pt` is only the point where
  generalisation peaked, and at 3M parameters that is the *least* useful model. Validation is
  opt-in for this reason (`--eval-interval 0` by default, which trains on the whole corpus).
- **The macOS/MPS job is absent from CI on purpose.** It failed repeatedly on the hosted
  runner. Do not reintroduce it without a reason; the MPS test above is the local substitute.
