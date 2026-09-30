import sys

import torch

from .model import MiniTransformer
from .vocab import decode, encode, vocab_size

if __name__ == "__main__":

    try:
        weights = torch.load(
            "model.pt",
            weights_only=True
        )
    except FileNotFoundError:
        sys.exit(
            "model.pt not found - "
            "run: python -m mini_transformer.train"
        )

    model = MiniTransformer(
        vocab_size=vocab_size,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=64
    )

    model.load_state_dict(weights)

    model.eval()

    prompt = "h"

    tokens = torch.tensor(
        [encode(prompt)],
        dtype=torch.long
    )

    generated = model.generate(
        tokens,
        max_new_tokens=20
    )

    print(decode(generated[0].tolist()))
