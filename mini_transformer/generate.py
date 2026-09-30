import argparse
import sys
from pathlib import Path

import torch

from .config import resolve_checkpoint_config
from .model import MiniTransformer
from .tokenizer import EOT, decode, encode, load_tokenizer
from .train import resolve_device


def load_checkpoint(path, device, vocab_size, config=None):
    """`config` is for callers holding an independent config to check against
    the checkpoint's; the CLI omits it, so the checkpoint's own config wins."""
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{path} not found - run: uv run python -m mini_transformer.train"
        )

    state = torch.load(path, map_location=device, weights_only=True)

    # Before the weight and vocabulary checks below, and before the model is
    # built: naming a differing hyperparameter is more useful than whatever
    # downstream error it would otherwise cause.
    config = resolve_checkpoint_config(state.get("config"), config, path)

    if "model" not in state or "token_embedding.weight" not in state["model"]:
        raise ValueError(
            f"{path} is not a mini-transformer checkpoint (no 'model' entry with a "
            f"token embedding) - point --checkpoint at a file written by "
            f"mini_transformer.train"
        )

    saved_vocab = state["model"]["token_embedding.weight"].shape[0]
    if saved_vocab != vocab_size:
        raise ValueError(
            f"checkpoint was trained on a {saved_vocab}-token vocabulary but "
            f"{vocab_size} tokens were requested - point --tokenizer at the "
            f"tokenizer used for training, or retrain the model"
        )

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
    p.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="top-k cutoff; 0 disables",
    )
    p.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="penalty for repeating tokens (> 1.0 penalizes)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="use KV-cache during generation for fast autoregressive decode",
    )
    p.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="stream tokens to stdout as they are generated",
    )
    p.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        default=False,
        help="run in an interactive prompt loop",
    )
    args = p.parse_args(argv)

    device = resolve_device(args.device)

    if not Path(args.tokenizer).exists():
        raise ValueError(
            f"{args.tokenizer} not found - run: uv run python -m mini_transformer.train"
        )

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()

    model, config = load_checkpoint(args.checkpoint, device, vocab_size)

    def generate_single_prompt(prompt_text, stream=args.stream):
        ids = encode(tokenizer, prompt_text)
        if not prompt_text.strip():
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

        tokens = torch.tensor([ids], dtype=torch.long, device=device)

        if stream:
            if prompt_text.strip():
                print(prompt_text, end="", flush=True)

            def stream_token(token_tensor):
                print(decode(tokenizer, token_tensor[0].tolist()), end="", flush=True)

            out = model.generate(
                tokens,
                max_new_tokens=args.n,
                config=config,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                use_cache=args.use_cache,
                on_token=stream_token,
            )
            print()
        else:
            out = model.generate(
                tokens,
                max_new_tokens=args.n,
                config=config,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                use_cache=args.use_cache,
            )
            print(decode(tokenizer, out[0].tolist()))

    if args.interactive:
        print("MiniTransformer interactive mode (Ctrl+C or 'exit' to quit)\n")
        while True:
            try:
                user_prompt = input("Prompt: ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_prompt.strip() in {"exit", "quit"}:
                break
            generate_single_prompt(user_prompt, stream=True)
            print()
    else:
        print(
            f"device={device} vocab={vocab_size} block_size={config.block_size} "
            f"temperature={args.temperature} top_p={args.top_p} n={args.n}",
            file=sys.stderr,
        )
        generate_single_prompt(args.prompt, stream=args.stream)


def cli():
    """Entry point for the `mini-transformer` console script.

    `main` raises `ValueError` for anything the user can fix (no checkpoint, wrong
    vocabulary) and the `__main__` guard turns that into a one-line message. A console
    script calls `main` directly, so it needs the same conversion.
    """
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    cli()
