text = "hello world"

chars = sorted(set(text))

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for ch, i in stoi.items()
}

vocab_size = len(chars)


def encode(text):
    return [stoi[ch] for ch in text]


def decode(ids):
    return "".join(
        itos[i]
        for i in ids
    )
