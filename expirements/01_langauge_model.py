import torch


text = "hello world"

chars = sorted(set(text))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)

print("Vocabulary:", chars)
print("Vocabulary size:", len(chars))

print("\nEncoded:")
print(data)

print("\nDecoded:")
print(decode(data.tolist()))