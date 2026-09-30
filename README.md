# mini-transformer

A ~3M-parameter decoder-only Transformer language model, written from scratch in
PyTorch and trained on real text. No `transformers`, no `nn.Transformer` — RMSNorm,
RoPE, SwiGLU, causal multi-head attention, the BPE tokenizer and nucleus sampling are
all here, readable, in 734 lines including the CLIs.

The point is legibility at a size you can actually watch train. It reaches a last-logged
loss of **0.090** (iteration 5900 of 6000 — the trainer only prints every 100 steps) in
**7 minutes 20 seconds** on an Apple Silicon laptop with the MPS backend, and then
reproduces a 200,000-character slice of WikiText-2 almost verbatim. That timing is an
MPS number; the CPU path is slower.

## Quickstart

```bash
uv sync
uv run python -m mini_transformer.train      # ~7 min: trains, writes checkpoints/model.pt
uv run python -m mini_transformer.generate    # prints sampled text
```

`mini-transformer` is installed as a console script pointing at the same entry point, so
after `uv sync` this works too:

```bash
uv run mini-transformer                      # same as python -m mini_transformer.generate
```

Running it before training exits with a message telling you to run the trainer, rather
than a placeholder greeting.

Requires Python 3.14+. Runtime dependencies are `torch`, `tokenizers`, `datasets` and
`numpy` (torch initialises NumPy at startup and warns without it).

## What the recorded run measured

Every number below was measured on one fresh `rm -rf data checkpoints && uv run python
-m mini_transformer.train`, not copied from a design doc. Reproduce it by re-running that
command: the loss trace should match to within floating-point noise (the trainer seeds
torch and the batch sampler, but MPS kernels do not promise bit-exact results), and the
wall clock will differ.

| | Measured |
|---|---|
| Training loss, last logged (step 5900 of 6000) | **0.090** |
| Wall clock for the whole run, dataset download included | **440 s (7 m 20 s)** |
| Teacher-forced next-token accuracy (training set, no held-out split) | **98.3 %** (40,244 / 40,960 tokens) |
| BPE vocabulary | **6,582** tokens (8,192 requested) |
| Parameters | **3,034,944** |
| Packed training corpus | **43,656** tokens (200,000 characters of WikiText-2) |
| Device | Apple MPS |

Loss is logged every 100 steps. This is a sampled subset of that trace, not every row —
chosen to include two of the steps where it rises rather than falls, because a table
sampled only at round thousands looks monotone and hides that:

```
iter      elapsed   loss
     0      0.4s    8.802
   100      7.1s    6.493
  1000     66.7s    1.581
  1300     88.9s    1.036
  1400     95.5s    1.569    <- up, not down
  2000    135.1s    0.670
  2800    188.4s    0.264
  2900    195.0s    0.304    <- up, not down
  3000    201.7s    0.248
  4000    273.0s    0.122
  5000    351.3s    0.113
  5900    424.0s    0.090
```

Per-batch loss on a 43,656-token corpus is noisy, so the real curve wobbles like this.
The trend is what matters, and the 0.090 at the end is not the minimum of a smooth
descent — it is one sample from the tail.

Verbatim output of `uv run python -m mini_transformer.generate --n 300` (default
prompt `"The "`, temperature 0.8, top-p 0.95, one long line each). This is one run's
capture: the sampler is not seeded, so re-running gives you different text of similar
quality, and the quality varies a lot between draws.

```
The 00 ft / Ikaki m ) long overall and had a long ventral fin fold of arrived in the seven @-@ class battleships , 6 @.@ 4 @-@ inch of heran . Most of the theme wasoser in two above water 45 @-@

 = = Milestones = =

 When Mason was injured in warm @-@ ups late in the year , Columbus was without an active goaltender on their roster . To remedy the situation , the team signed former University of Michigan goaltender Shawn Hunwick to a one @-@ day , amateur tryout contract . After being eliminated from the NCAA Tournament just days prior , Hunwick skipped an astronomy class and drove his worn down 2003 Ford Ranger to Columbus to make the game . He served as the back @-@ up to Allen York during the game , and the following day , he signed a contract for the remainder of the year . With Mason returning from injury , Hunwick was third on the team 's depth chart when an injury to York allowed Hunwick to remain as the back @-@ up for the final two games of the year . In the final game of the season , the Blue Jackets were leading the Islanders 7 – 3 with 2 : 33 remaining when , at the behest of his teammates , Head Coach Todd Richards put Hunwick in to finish the game . He did not face a shot . Hunwick was the franchise record ninth player to make his NHL debut during the season . Conversely , Vaclav Prosp
```

Read that carefully, because it is the whole project in one paragraph. The first line is
noise: the model has no memory of the prompt `"The "` starting here, so it guesses and
lands on a warship. It then hits a ` = = Milestones = = ` section header, and from that
header onwards it **reproduces the WikiText-2 article about Shawn Hunwick almost word
for word** — dates, names, scores, quotation marks and all. That is memorisation of a
43,656-token corpus, not language modelling of English.

`@.@` and `@-@` are literal characters in WikiText-2's preprocessed text, not model
output artefacts. Once it latches onto a memorised passage the text is fluent, but
before that it is noise — that is the memorisation caveat, in one sample.

## Architecture

Decoder-only, **pre-norm**, with rotary position embeddings:

```
tokens ──▶ token_embedding (tied to lm_head) ──▶ x
                                                │
                            ┌───────────────────┴───────────────────┐
                            │  TransformerBlock × 4                │
                            │    x = x + MHA(RMSNorm(x))            │
                            │    x = x + SwiGLU(RMSNorm(x))         │
                            │  MHA applies RoPE to q, k            │
                            │  is_causal=True, so no looking ahead  │
                            └───────────────────────────────────────┘
                                                │
                                                ▼
                                       RMSNorm ──▶ lm_head (tied) ──▶ logits
```

Every layer lives in its own file, and the three that make up the model itself carry a
`__main__` demo you can run on a random tensor:

```bash
uv run python -m mini_transformer.transformer_block
uv run python -m mini_transformer.multi_head_attention
uv run python -m mini_transformer.model
```

The rest either support the model (config, tokenizer, data, sampling, init) or are single
layers from the diagram with no demo attached — `python -m mini_transformer.rmsnorm`
exits silently. For RoPE, RMSNorm and SwiGLU, `expirements/03_rope.py`, `04_rmsnorm.py` and
`05_swiglu.py` walk through each concept, importing the implementation from the package
rather than reimplementing it: useful as a worked example of the call sequence, not as a
pre-package build.

Parameter arithmetic for `d_model=192`, `d_ff=512`, `n_layer=4`, `vocab=6,582`:

| Component | Count |
|---|---|
| Token embedding (`lm_head` is tied, so it adds nothing) | 6582 × 192 = 1,263,744 |
| One block: 2 × RMSNorm (384) + QKV (110,592) + output proj (36,864) + SwiGLU (294,912) | 442,752 |
| Four blocks | 1,771,008 |
| Final RMSNorm | 192 |
| **Total** | **3,034,944** |

The design spec quotes 3,344,064 parameters for this architecture, but that figure was
measured at a *requested* vocabulary of 8,192. This corpus only yields 6,582 merges, and
the head is sized from the real value, so the shipped model is smaller.
`train_tokenizer` prints a warning when that happens.

## Configuration

All hyperparameters live in `config.py`; `train.py` and `generate.py` read them and the
checkpoint stores a copy, so a mismatched checkpoint is rejected rather than silently
loaded.

Only six of the twenty fields have a CLI flag (`--dataset`, `--dataset-config`,
`--corpus-chars`, `--iters`, `--batch-size`, `--lr`). **A hyperparameter sweep over any
of the other fourteen means editing `config.py`** — there is no `--config` file flag.
`--resume` deliberately ignores `iters`, since raising the iteration count is what it is
for; every other field changing mid-run is rejected.

| Data | Value |
|---|---|
| Dataset | `Salesforce/wikitext`, config `wikitext-2-raw-v1` |
| Corpus slice | first 200,000 characters of the train split, blank rows dropped |
| Tokenizer | byte-level BPE, 8,192 requested |
| Packing | `uint16` (`tokenizer.py` refuses a vocabulary above 65,535) |

| Model | Value |
|---|---|
| `d_model` | 192 |
| `n_layer` | 4 |
| `n_head` | 6 (head dim 32) |
| `d_ff` | 512 |
| `block_size` | 128 |
| Weight tying | `lm_head.weight is token_embedding.weight` |

| Optimisation | Value |
|---|---|
| Iterations | 6,000 |
| Batch size | 8 |
| Optimiser | AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.1` on 2-D params only |
| LR schedule | 100 warmup steps to `6e-3`, cosine decay to `6e-4` |
| Gradient clipping | 1.0 |
| Initialisation | `N(0, 0.02)`, residual projections at `0.02 / sqrt(2 * n_layer)` |
| Seed | 42 (torch and the batch sampler) |

## Generation

```bash
uv run python -m mini_transformer.generate --n 300 --temperature 0.8 --top-p 0.95
uv run python -m mini_transformer.generate --prompt "The museum " --n 200
uv run python -m mini_transformer.generate --temperature 0.0   # greedy
```

From Python:

```python
model.generate(tokens, max_new_tokens=200)                              # sampled, temperature=1.0
model.generate(tokens, max_new_tokens=200, temperature=0.8, top_p=0.95)  # nucleus
model.generate(tokens, max_new_tokens=200, temperature=0.0)             # greedy, deterministic
```

`temperature=0.0` short-circuits to `argmax` and is the only deterministic setting —
the defaults (`temperature=1.0, top_p=1.0`) sample the full untruncated distribution, so
two calls with the same tokens give different text. Above zero, the top-p tail is dropped
and the sample is drawn from the renormalised nucleus; `sampling.py` maps the position
returned by `torch.multinomial` back to a vocabulary id through the sort permutation,
which is the step that is easy to omit and hard to spot when it is missing.

## What this model is, and what it is not

**It memorises its training corpus. It does not learn general English.** The entire
training set is 43,656 tokens from 200,000 characters of WikiText-2. At 3M parameters
trained for 6,000 steps, the model reaches a loss of 0.090 by reproducing that slice
largely verbatim, as the sample above shows. That is exactly what makes a 7-minute
laptop run produce readable text, and it is not evidence of language understanding. Do
not read the 98.3 % accuracy as generalisation: **there is no held-out set at all.**
`get_batch` samples uniformly from the whole packed corpus for both training and any
evaluation, so every number in this README is a training-set number.

Also worth knowing before you read generated output:

- **`<|endoftext|>` is never trained, and nothing stops on it.** The packed corpus
  contains zero end tokens — `data.py` joins rows with `"\n\n"` and never inserts one —
  so that embedding row never receives a gradient and the model is never taught to
  produce the token; `generate` also has no stop-on-EOT path. Output always runs the
  full `--n` tokens, and `decode` strips the token if it ever appears. A sample that
  runs on past where you expected a stop is not the model failing to learn to stop: it
  was never taught to.
- **Punctuation is spaced** (` , ` `. `) and some tokens are byte-level fragments, because
  that is how the corpus is written.
- **Prompts the model has no memorised continuation for start as noise.** A prompt that
  does not appear in the corpus may produce a garbage first line before the model finds a
  passage it knows, which is what the sample above does.

## Known ceilings

- **No KV cache.** `generate()` re-runs the entire window on every step, so decoding is
  **O(n²)** in sequence length: `n` new tokens cost `n` full-window forwards rather than
  `n` single-token forwards. Deliberate at this scale, where the whole context is 128
  tokens. The upgrade path is a per-layer cache of `k` and `v` with one row appended per
  step; the attention call already takes `cos`/`sin` per position, so the change is
  confined to `multi_head_attention.py` and the loop in `model.generate`.
- **Single-sequence decode.** `generate()` handles one prompt at a time.
- **No train/validation split** and no checkpoint selection: the last step is the
  deliverable. Use `--resume` to extend a run, which is how you would get a validation
  number if you added one.
- **`uint16` packing caps the vocabulary at 65,535 tokens.** `train_tokenizer` raises
  rather than silently truncating ids.

## Layout

```
mini_transformer/
  config.py             every hyperparameter, JSON-serialisable
  tokenizer.py          BPE training, the uint16 guard, encode/decode
  data.py               corpus load, uint16 packing, batch sampling
  model.py              MiniTransformer: embedding, block stack, tied head, generate()
  transformer_block.py  one pre-norm block
  multi_head_attention.py  QKV, head split, RoPE, causal SDPA
  feed_forward.py       SwiGLU
  rmsnorm.py            RMSNorm
  rope.py               rotary position embeddings
  sampling.py           temperature + nucleus sampling
  init.py               GPT-2 style initialisation
  train.py              training CLI: warmup, cosine decay, decoupled decay groups
  generate.py           sampling CLI and checkpoint validation
expirements/            teaching scripts, superseded by the package
tests/                  pytest suite
```

`expirements/` holds the step-by-step builds the package grew out of. The two earliest —
`01_langauge_model.py` (a bigram next-token model) and `02_self_attention.py` (raw-tensor
attention, before it became a module) — are self-contained and need only torch. The three
later ones, `03_rope.py`, `04_rmsnorm.py` and `05_swiglu.py`, import the shipped
implementations from `mini_transformer`, so they demonstrate the package and need it
installed.

## Development

```bash
uv run pytest -q           # 91 tests
uv run ruff check .
uv build                   # wheel + sdist
```

## License

MIT — see [LICENSE](LICENSE).
