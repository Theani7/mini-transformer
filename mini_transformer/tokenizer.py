from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, Tokenizer

EOT = "<|endoftext|>"
MAX_UINT16 = 65535


def train_tokenizer(texts, vocab_size, path):
    if vocab_size > MAX_UINT16:
        raise ValueError(
            f"vocab_size {vocab_size} exceeds {MAX_UINT16}; "
            "ids are packed as uint16 and would be silently corrupted"
        )
    if not any(text.strip() for text in texts):
        raise ValueError("cannot train a tokenizer on an empty or whitespace-only corpus")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(texts, vocab_size=vocab_size, special_tokens=[EOT])

    actual = tokenizer.get_vocab_size()
    if actual < vocab_size:
        print(
            f"warning: requested vocab_size={vocab_size} but the corpus only "
            f"yielded {actual} merges. The model head is sized from the real value."
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return actual


def load_tokenizer(path):
    return Tokenizer.from_file(str(path))


def encode(tokenizer, text):
    return tokenizer.encode(text).ids


def decode(tokenizer, ids):
    return tokenizer.decode(ids)
