from pathlib import Path

import numpy as np
import torch

from .tokenizer import encode


def load_corpus(dataset, dataset_config, corpus_chars):
    from datasets import load_dataset

    splits = load_dataset(dataset, dataset_config)
    text = "\n\n".join(row for row in splits["train"]["text"] if row.strip())

    if not text:
        raise ValueError(
            f"corpus from {dataset}/{dataset_config} is empty after filtering "
            "blank rows; nothing to train on"
        )

    return text[:corpus_chars]


def pack_ids(tokenizer, text, path):
    ids = np.array(encode(tokenizer, text), dtype=np.uint16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(ids.tobytes())
    return len(ids)


def load_packed(path, block_size):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    if len(data) <= block_size:
        raise ValueError(
            f"corpus has {len(data)} tokens but block_size is {block_size}; "
            "need at least block_size + 1 tokens to form a batch"
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
