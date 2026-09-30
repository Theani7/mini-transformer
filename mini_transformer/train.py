import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config, resolve_checkpoint_config
from .data import get_batch, load_corpus, load_packed, pack_ids
from .init import init_weights
from .model import MiniTransformer
from .tokenizer import EOT, decode, encode, load_tokenizer, train_tokenizer

CACHE_KEY = ("dataset", "dataset_config", "corpus_chars", "bpe_vocab_size")


def resolve_device(requested):
    if requested != "auto":
        return requested
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
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--resume", action="store_true")
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
    )

    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    torch.manual_seed(config.seed)

    tokenizer_path = data_dir / "tokenizer.json"
    train_path = data_dir / "train.bin"
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

    stale = stamp is None or _cache_key(stamp) != _cache_key(config)
    if stale or not tokenizer_path.exists() or not train_path.exists():
        text = load_corpus(config.dataset, config.dataset_config, config.corpus_chars)
        train_tokenizer([text], config.bpe_vocab_size, tokenizer_path)
        count = pack_ids(load_tokenizer(tokenizer_path), text, train_path)
        config.save(stamp_path)
        print(f"packed {count:,} tokens -> {train_path}")

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"tokenizer: {vocab_size} tokens (requested {config.bpe_vocab_size})")

    data = load_packed(train_path, config.block_size)

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
            state.get("config"), config, out_path, ignore={"iters"}
        )
        try:
            model.load_state_dict(state["model"])
        except RuntimeError as exc:
            raise ValueError(
                f"checkpoint weights do not fit the requested config: {exc}"
            ) from exc

    optimizer = build_optimizer(model, config)
    generator = torch.Generator().manual_seed(config.seed)

    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}")
    print("iter      elapsed   loss")

    started = time.perf_counter()
    completed = start

    for step in range(start, config.iters):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(config, step)

        x, y = get_batch(
            data, config.batch_size, config.block_size, device, generator
        )
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
            print(f"{step:>6}  {elapsed:>7.1f}s  {loss.item():>7.3f}", flush=True)

        if config.sample_interval and step and step % config.sample_interval == 0:
            print(_sample(model, tokenizer, config, device, ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": completed, "config": config.__dict__}, out_path)
    print(f"\nsaved {out_path}")


def _cache_key(config):
    return tuple(getattr(config, field) for field in CACHE_KEY)


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
