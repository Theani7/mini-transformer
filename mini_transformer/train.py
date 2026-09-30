import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_model

from .config import Config, resolve_checkpoint_config
from .data import get_batch, load_corpus, load_packed, pack_ids, split_ids
from .init import init_weights
from .model import MiniTransformer
from .tokenizer import EOT, decode, encode, load_tokenizer, train_tokenizer

CACHE_KEY = (
    "dataset",
    "dataset_config",
    "corpus_chars",
    "bpe_vocab_size",
)


def resolve_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def lr_at(config, step):
    if step < config.warmup:
        return config.lr * step / config.warmup
    progress = (step - config.warmup) / max(config.iters - config.warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.lr * (config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine)


def build_optimizer(model, config):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=(config.beta1, config.beta2),
    )


def evaluate(model, data, batch_size, block_size, device):
    """Mean loss over every window in the split.

    Exhaustive rather than sampled: a 5% slice of a 43k-token corpus is only a
    few hundred windows, and covering all of them makes the number
    deterministic and low-variance enough to select a checkpoint on.
    """
    was_training = model.training
    model.eval()

    total = len(data) - block_size - 1
    cos, sin = _rope(model.config, device)
    losses = []

    with torch.no_grad():
        for start in range(0, total, batch_size):
            count = min(batch_size, total - start)
            indices = torch.arange(start, start + count)
            x = torch.stack(
                [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in indices]
            )
            y = torch.stack(
                [
                    torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
                    for i in indices
                ]
            )
            logits = model(x.to(device), cos, sin)
            losses.append(
                F.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    y.to(device).view(-1),
                    reduction="sum",
                )
            )

    if was_training:
        model.train()

    # summed over tokens, so divide by tokens and not by windows
    return float(torch.stack(losses).sum() / (total * block_size))


def parse_args(argv=None):
    defaults = Config()
    p = argparse.ArgumentParser(description="Train the mini transformer")
    p.add_argument("--dataset", default=defaults.dataset)
    p.add_argument("--dataset-config", default=defaults.dataset_config)
    p.add_argument("--corpus-chars", type=int, default=defaults.corpus_chars)
    p.add_argument("--iters", type=int, default=defaults.iters)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--val-fraction",
        type=float,
        default=defaults.val_fraction,
        help="fraction of tokens held out (only used when --eval-interval > 0)",
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=defaults.eval_interval,
        help="steps between validation evals; 0 (default) disables validation "
        "and trains on the whole corpus",
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop if val loss fails to improve for N consecutive evals; 0 disables",
    )
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--best-out", default="checkpoints/best.pt")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="no",
        help="enable mixed precision training (no, fp16, bf16)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = Config(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        corpus_chars=args.corpus_chars,
        iters=args.iters,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        eval_interval=args.eval_interval,
    )

    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    torch.manual_seed(config.seed)

    tokenizer_path = data_dir / "tokenizer.json"
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    stamp_path = data_dir / "config.json"

    # The stamp is a cache, written only by this module, so a format change
    # should invalidate it rather than crash. `Config.load` stays strict for
    # every other caller. `ValueError` covers the other half of "format change":
    # `Config.save` is a non-atomic `write_text`, so a crash mid-write leaves
    # truncated JSON and `json.loads` raises `JSONDecodeError`.
    try:
        stamp = Config.load(stamp_path) if stamp_path.exists() else None
    except (TypeError, ValueError):
        stamp = None

    needs_val = bool(config.eval_interval)
    stale = stamp is None or _cache_key(stamp) != _cache_key(config)
    if (
        stale
        or not tokenizer_path.exists()
        or not train_path.exists()
        or (needs_val and not val_path.exists())
    ):
            text = load_corpus(
                config.dataset, config.dataset_config, config.corpus_chars
            )
            train_tokenizer([text], config.bpe_vocab_size, tokenizer_path)
            tokenizer = load_tokenizer(tokenizer_path)

            ids = np.array(encode(tokenizer, text), dtype=np.uint16)
            if config.eval_interval:
                # Only hold out a val set if something will score it. With eval
                # off, discarding 5% of the corpus buys nothing.
                train_ids, val_ids = split_ids(
                    ids, config.val_fraction, config.block_size
                )
                n_train = pack_ids(train_ids, train_path)
                n_val = pack_ids(val_ids, val_path)
                packed = f"packed {n_train:,} train / {n_val:,} val tokens"
            else:
                n_train = pack_ids(ids, train_path)
                packed = f"packed {n_train:,} train tokens (no val split)"

            config.save(stamp_path)
            print(packed)

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"tokenizer: {vocab_size} tokens (requested {config.bpe_vocab_size})")

    data = load_packed(train_path, config.block_size)
    val_data = (
        load_packed(val_path, config.block_size) if config.eval_interval else None
    )

    model = MiniTransformer(vocab_size=vocab_size, config=config)
    init_weights(model, config.n_layer)
    model = model.to(device)

    start = 0
    out_path = Path(args.out)
    state = None
    if args.resume and out_path.exists():
        state = torch.load(out_path, map_location=device, weights_only=True)
        start = state["step"]
        print(f"resumed from step {start}")

    # After the early return on purpose. A checkpoint that already holds the
    # requested step count is reported as finished, whatever flags this run was
    # given: passing a *lower* --iters than the one it was trained under is a
    # legitimate no-op, and the strict diff below would otherwise reject it.
    if start >= config.iters:
        print(f"nothing to do: {out_path} already holds step {start}")
        return

    if state is not None:
        # Before the load, not after: naming the differing hyperparameter beats
        # reporting a size mismatch it caused. `generate` checks in the same
        # order, which is why both loaders share `resolve_checkpoint_config`.
        # `iters` is ignored because raising it is what `--resume` is for; every
        # other field silently changing mid-run is the bug this closes.
        resolve_checkpoint_config(
            state.get("config"),
            config,
            out_path,
            ignore={"iters", "eval_interval"},
        )
        if config.eval_interval and not state["config"].get("eval_interval"):
            print(
                "warning: validation turned on for a resume, but the val tail "
                "was in the training data up to this step - the number is "
                "contaminated, not a held-out score"
            )
        try:
            model.load_state_dict(state["model"])
        except RuntimeError as exc:
            raise ValueError(
                f"checkpoint weights do not fit the requested config: {exc}"
            ) from exc

    optimizer = build_optimizer(model, config)
    if state is not None and "optimizer" in state:
        try:
            optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"warning: could not restore optimizer state: {exc}")
    generator = torch.Generator().manual_seed(config.seed)

    dev_type = device.type if isinstance(device, torch.device) else str(device)
    amp_enabled = args.mixed_precision != "no" and dev_type in ("cuda", "cpu", "mps")
    amp_dtype = (
        torch.bfloat16
        if args.mixed_precision == "bf16"
        else (torch.float16 if args.mixed_precision == "fp16" else torch.float32)
    )
    if amp_enabled:
        print(f"mixed_precision={args.mixed_precision} ({amp_dtype}) on {dev_type}")

    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")
    print("iter      elapsed     train      val   best" if config.eval_interval
          else "iter      elapsed     train")

    started = time.perf_counter()
    completed = start
    val_loss = None
    # Restored, not reset. Without this a resume starts with `best_val = None`,
    # so the first eval unconditionally overwrites a better `best.pt` that the
    # previous run already wrote - the selection is silently destroyed.
    best_val = state.get("best_val") if state else None
    best_step = state.get("best_step", start) if state else start
    best_path = Path(args.best_out)
    if config.eval_interval:
        # The first eval fires inside the loop, before the final save creates
        # the directory, so the parent has to exist up front.
        best_path.parent.mkdir(parents=True, exist_ok=True)

    patience_counter = 0
    for step in range(start, config.iters):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(config, step)

        x, y = get_batch(
            data, config.batch_size, config.block_size, device, generator
        )
        with torch.autocast(device_type=dev_type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(x, *_rope(config, device))
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1)
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        completed = step + 1

        if step % 100 == 0:
            elapsed = time.perf_counter() - started
            val_text = f"{val_loss:>8.3f}" if val_loss is not None else " " * 8
            best_text = (
                f"{best_val:>7.3f}" if config.eval_interval and best_val is not None else " " * 7
            )
            print(
                f"{step:>6}  {elapsed:>7.1f}s  {loss.item():>8.3f}"
                f"{val_text}{best_text}",
                flush=True,
            )

        if needs_val and step and step % config.eval_interval == 0:
            val_loss = evaluate(
                model, val_data, config.batch_size, config.block_size, device
            )
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                best_step = completed
                patience_counter = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": completed,
                        "val_loss": val_loss,
                        "config": config.__dict__,
                    },
                    best_path,
                )
                save_model(
                    model,
                    best_path.with_suffix(".safetensors"),
                    metadata={
                        "config": json.dumps(config.__dict__),
                        "step": str(completed),
                        "val_loss": str(best_val),
                    },
                )
            else:
                patience_counter += 1
                if (
                    args.early_stopping_patience
                    and patience_counter >= args.early_stopping_patience
                ):
                    print(
                        f"early stopping at step {completed}: val loss failed to improve "
                        f"for {patience_counter} consecutive evals"
                    )
                    break

        if config.sample_interval and step and step % config.sample_interval == 0:
            print(_sample(model, tokenizer, config, device, ""))

    if needs_val and best_val is None:
        val_loss = evaluate(
            model, val_data, config.batch_size, config.block_size, device
        )
        best_val, best_step = val_loss, completed
        torch.save(
            {
                "model": model.state_dict(),
                "step": completed,
                "val_loss": val_loss,
                "config": config.__dict__,
            },
            best_path,
        )
        save_model(
            model,
            best_path.with_suffix(".safetensors"),
            metadata={
                "config": json.dumps(config.__dict__),
                "step": str(completed),
                "val_loss": str(best_val),
            },
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": completed,
            "config": config.__dict__,
            # Carried so a later --resume knows what best already means.
            "best_val": best_val,
            "best_step": best_step,
        },
        out_path,
    )
    safetensors_path = out_path.with_suffix(".safetensors")
    save_model(
        model,
        safetensors_path,
        metadata={"config": json.dumps(config.__dict__), "step": str(completed)},
    )
    print(f"\nsaved {out_path} and {safetensors_path}")

    if best_val is not None:
        source = "best" if config.eval_interval else "best (from the resumed run)"
        ppl = math.exp(best_val)
        print(
            f"{source} val {best_val:.3f} (ppl {ppl:.2f}) at step {best_step} -> {best_path}"
        )


def _cache_key(config):
    """Identity of the *packed bytes*, not of the run.

    Whether a val set was held out changes what lands in `train.bin`, so the
    stamp has to record that. The eval cadence does not touch the bytes, so
    retuning `--eval-interval` must not cost a retokenize.
    """
    split = (config.val_fraction, True) if config.eval_interval else (None, False)
    return tuple(getattr(config, field) for field in CACHE_KEY) + (split,)


def _rope(config, device):
    from .rope import rope_cache

    return rope_cache(config.block_size, config.head_dim, device=device)


def _sample(model, tokenizer, config, device, prompt):
    model.eval()
    ids = encode(tokenizer, prompt) or [tokenizer.token_to_id(EOT)]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(tokens, config.block_size, config, temperature=0.8, top_p=0.95)
    model.train()
    return decode(tokenizer, out[0].tolist())


if __name__ == "__main__":
    main()
