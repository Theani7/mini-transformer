import torch


def rope_cache(seq_len, head_dim, device=None, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)

    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    if x.dim() != 4 or x.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"apply_rope expects (batch, heads, seq, head_dim); got shape "
            f"{tuple(x.shape)} against a cache with head_dim {cos.shape[-1]}"
        )
    return x * cos + rotate_half(x) * sin
