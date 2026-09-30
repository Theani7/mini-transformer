import argparse
import math
import sys
from pathlib import Path

import numpy as np

from .data import load_packed
from .generate import load_checkpoint
from .tokenizer import encode, load_tokenizer
from .train import evaluate, resolve_device


def evaluate_text(model, tokenizer, text, batch_size, block_size, device):
    ids = np.array(encode(tokenizer, text), dtype=np.uint16)
    if len(ids) < block_size + 2:
        raise ValueError(
            f"text is {len(ids)} tokens; need at least {block_size + 2} tokens to evaluate"
        )
    return evaluate(model, ids, batch_size, block_size, device)


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on data or text")
    default_ckpt = (
        "checkpoints/model.safetensors"
        if Path("checkpoints/model.safetensors").exists()
        else "checkpoints/model.pt"
    )
    default_tok = (
        "checkpoints/tokenizer.json"
        if Path("checkpoints/tokenizer.json").exists()
        else "data/tokenizer.json"
    )
    p.add_argument("--checkpoint", default=default_ckpt)
    p.add_argument("--tokenizer", default=default_tok)
    p.add_argument("--data", help="path to packed uint16 .bin file (e.g. data/val.bin)")
    p.add_argument("--text-file", help="path to a plaintext file to evaluate")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    if not args.data and not args.text_file:
        args.data = "data/val.bin"

    device = resolve_device(args.device)

    if not Path(args.tokenizer).exists():
        raise ValueError(f"{args.tokenizer} not found")

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    model, config = load_checkpoint(args.checkpoint, device, vocab_size)

    if args.text_file:
        text_path = Path(args.text_file)
        if not text_path.exists():
            raise ValueError(f"text file {text_path} not found")
        text = text_path.read_text()
        loss = evaluate_text(model, tokenizer, text, args.batch_size, config.block_size, device)
        source = f"text {text_path}"
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            raise ValueError(f"data file {data_path} not found")
        data = load_packed(data_path, config.block_size)
        loss = evaluate(model, data, args.batch_size, config.block_size, device)
        source = f"data {data_path}"

    ppl = math.exp(loss)
    print(f"[{source}] loss: {loss:.4f}  ppl: {ppl:.2f}")
    return loss, ppl


def cli():
    try:
        main()
    except ValueError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    cli()
