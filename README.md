# mini-transformer

A character-level Transformer built from scratch in PyTorch. No `transformers`, no
`nn.Transformer` — every layer is written out so the mechanics are visible:
multi-head attention, the causal mask, positional embeddings, the residual stream.

The point is legibility, not scale. It trains on eleven characters.

## Quickstart

```bash
uv sync
uv run python -m mini_transformer.train      # trains, writes model.pt
uv run python -m mini_transformer.generate   # prints generated text
```

```
step    0 | loss 2.5667
step  100 | loss 0.0491
step  900 | loss 0.0015

saved model.pt
```

```
$ uv run python -m mini_transformer.generate
hello worldorllllllo 
```

It reproduces `hello world` exactly, then emits noise — the training set is that one
string, so nothing follows the last `d`. `train.py` seeds torch at 42 and generation is
greedy, so both numbers above are reproducible run to run.

Requires Python 3.14+. Runtime dependencies are `torch` and `numpy` — the code imports
only `torch`, but torch initializes NumPy at startup and warns without it.

## Architecture

Decoder-only, post-LayerNorm, following the original 2017 layout:

```
tokens ──▶ token_embedding ┐
                           ├─▶ x
positions ─▶ position_emb ┘
                │
                ▼
        ┌── TransformerBlock ×2 ──┐
        │  x = x + MHA(LayerNorm(x))   ← causal, prevents looking ahead
        │  x = x + FFN(LayerNorm(x))   ← Linear → GELU → Linear
        └──────────────────────────────┘
                │
                ▼
          LayerNorm → Linear → logits
```

| Setting | Value | Where |
|---|---|---|
| `d_model` | 32 | `train.py` |
| `num_heads` | 4 | `train.py` |
| `d_ff` | 128 | `train.py` |
| `num_layers` | 2 | `train.py` |
| `max_seq_len` | 64 | `train.py` |
| optimizer | AdamW, `lr=1e-3` | `train.py` |
| steps | 1000 | `train.py` |
| batch size | 1 | `train.py` |
| seed | 42 | `train.py` |

Positions are a learned `nn.Embedding`, not sinusoidal — the nanoGPT choice, and the
better one at this size.

## Layout

```
mini_transformer/
  model.py               MiniTransformer: embeddings, block stack, generate()
  transformer_block.py   one post-LN block
  multi_head_attention.py  Q/K/V projections, head split, causal mask, scaled dot-product
  feed_forward.py        Linear → GELU → Linear
  embeddings.py          bare nn.Embedding demo
  vocab.py               char ↔ id mapping, shared by train and generate
  train.py               overfits "hello world", saves model.pt
  generate.py            loads model.pt, decodes from a prompt
expirements/             standalone teaching scripts, not part of the package
tests/                   pytest suite
```

`expirements/` holds step-by-step builds that the package supersedes: a bigram
next-character model, then raw-tensor attention before it became a module. They are
kept runnable and independent of the package.

## Generation

`generate()` is greedy by default, which is why the output above is deterministic:

```python
model.generate(tokens, max_new_tokens=20)               # greedy
model.generate(tokens, max_new_tokens=20, temperature=0.8)  # sampled
```

## Limitations

Deliberate, and worth knowing before you build on this:

- **No KV cache.** `generate()` re-runs the whole window each step, so decoding is
  O(n²) in sequence length. Fine at 64 tokens; a real LM needs per-layer key/value
  caching.
- **No batching.** One sequence per step.
- **Hyperparameters are hardcoded** in `train.py` and `generate.py`. They must stay in
  sync — there is no config, no CLI flags, and the two files repeat the model
  constructor.
- **The dataset is a literal string** in `vocab.py`. `train.py` trains on the whole
  string as one sample, so there is no train/val split and no batching to split.
- **No `nn.Linear` weight tying** between `lm_head` and the token embedding.

## Development

```bash
uv run pytest              # 4 tests
uv run ruff check .        # lint
uv build                   # wheel + sdist
```

## License

MIT — see [LICENSE](LICENSE).
