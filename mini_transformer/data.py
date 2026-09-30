from pathlib import Path

import numpy as np
import torch


def load_corpus(dataset, dataset_config, corpus_chars):
    from datasets import load_dataset

    # A bare "wikitext" is not a valid namespace/name; the Hub retries it and the error takes 25 min.
    splits = load_dataset(dataset, dataset_config)
    text = "\n\n".join(row for row in splits["train"]["text"] if row.strip())[:corpus_chars]

    if not text:
        raise ValueError(
            f"corpus from {dataset}/{dataset_config} is empty after filtering blank "
            f"rows or slicing to corpus_chars={corpus_chars}; nothing to train on"
        )

    return text


def pack_ids(ids, path):
    """Write an id array to disk. Encoding and splitting are the caller's job,
    so the same primitive serves both the train and the val half."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(np.asarray(ids, dtype=np.uint16).tobytes())
    return len(ids)


def split_ids(ids, val_fraction, block_size):
    """Hold out the tail for validation.

    Both halves must be at least `block_size + 2` tokens or `load_packed`
    cannot form a batch from them, so the requested fraction is treated as a
    minimum, not an exact target.
    """
    if len(ids) == 0:
        raise ValueError("cannot split an empty corpus")

    minimum = block_size + 2
    n_val = int(len(ids) * val_fraction)
    if n_val < 1:
        raise ValueError(
            f"val_fraction={val_fraction} leaves no validation tokens "
            f"from {len(ids)}"
        )

    if len(ids) < 2 * minimum:
        raise ValueError(
            f"{len(ids)} tokens cannot be split into two sides of at least "
            f"{minimum}; raise corpus_chars or lower block_size"
        )

    n_val = max(n_val, minimum)
    cut = len(ids) - n_val
    return ids[:cut], ids[cut:]


def load_packed(path, block_size):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) < block_size + 2:
        raise ValueError(
            f"corpus has {len(data)} tokens but block_size is {block_size}; "
            "need at least block_size + 2 tokens to form a batch"
        )
    return data


def get_batch(data, batch_size, block_size, device, generator):
    high = len(data) - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)

    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )

    return x.to(device), y.to(device)
