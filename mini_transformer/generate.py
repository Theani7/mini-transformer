import argparse
import sys
from pathlib import Path

import torch

from .config import Config
from .model import MiniTransformer
from .tokenizer import EOT, decode, encode, load_tokenizer


def load_checkpoint(path, device, vocab_size, config=None):
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{path} not found - run: uv run python -m mini_transformer.train"
        )

    state = torch.load(path, map_location=device, weights_only=True)

    if config is None:
        config = Config(**state["config"]) if "config" in state else Config()

    saved = state.get("config")
    if saved is not None:
        differing = {
            key: (saved[key], getattr(config, key))
            for key in saved
            if key in config.__dict__ and saved[key] != getattr(config, key)
        }
        if differing:
            named = ", ".join(
                f"{k} (checkpoint {old!r}, requested {new!r})"
                for k, (old, new) in sorted(differing.items())
            )
            raise ValueError(f"checkpoint config differs: {named}")

    model = MiniTransformer(vocab_size=vocab_size, config=config)

    try:
        model.load_state_dict(state["model"])
    except RuntimeError as exc:
        raise ValueError(
            f"checkpoint weights do not fit the requested config: {exc}"
        ) from exc

    return model.to(device).eval(), config


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample from a trained checkpoint")
    p.add_argument("--checkpoint", default="checkpoints/model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--prompt", default="The ")
    p.add_argument("--n", type=int, default=200)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature; 0.0 gives greedy decoding",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="nucleus cutoff; 1.0 keeps the whole distribution",
    )
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    from .train import resolve_device

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    model, config = load_checkpoint(args.checkpoint, device, vocab_size)

    ids = encode(tokenizer, args.prompt)
    if not args.prompt.strip():
        print(
            f"blank prompt: seeding with {EOT} and sampling unconditionally",
            file=sys.stderr,
        )
        ids = [tokenizer.token_to_id(EOT)]

    if len(ids) > config.block_size:
        print(
            f"warning: prompt is {len(ids)} tokens; only the last "
            f"{config.block_size} reach the model",
            file=sys.stderr,
        )

    print(
        f"device={device} vocab={vocab_size} block_size={config.block_size} "
        f"temperature={args.temperature} top_p={args.top_p} n={args.n}",
        file=sys.stderr,
    )

    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        tokens,
        max_new_tokens=args.n,
        config=config,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(decode(tokenizer, out[0].tolist()))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))
