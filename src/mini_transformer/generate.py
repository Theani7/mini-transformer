import torch

from src.model import MiniTransformer


text = "hello world"

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


vocab_size = len(chars)

model = MiniTransformer(
    vocab_size=vocab_size,
    d_model=32,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    max_seq_len=64,
)

# TODO: load trained parameters here

model.eval()

prompt = "h"

tokens = torch.tensor(
    [encode(prompt)],
    dtype=torch.long,
)

generated = model.generate(
    tokens,
    max_new_tokens=10,
)

print(decode(generated[0].tolist()))