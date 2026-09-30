import torch
import math


def positional_encoding(seq_len, embedding_dim):
    position = torch.arange(seq_len).unsqueeze(1)

    encoding = torch.zeros(seq_len, embedding_dim)

    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2)
        * (-math.log(10000.0) / embedding_dim)
    )

    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)

    return encoding


seq_len = 8
embedding_dim = 4

pe = positional_encoding(seq_len, embedding_dim)

print(pe)
print("\nShape:", pe.shape)