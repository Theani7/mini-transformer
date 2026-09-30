import torch
import torch.nn.functional as F
from torch import nn

from .rope import apply_rope, rope_cache


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                f"d_model {d_model} is not divisible by n_head {n_head}"
            )

        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        batch_size, seq_len, d_model = x.shape

        q, k, v = self.qkv(x).split(d_model, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos[:, :, :seq_len], sin[:, :, :seq_len])
        k = apply_rope(k, cos[:, :, :seq_len], sin[:, :, :seq_len])

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.proj(out)


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(1, 16, 32)
    cos, sin = rope_cache(16, 8)

    attention = MultiHeadAttention(d_model=32, n_head=4)

    print("Input shape:", x.shape)
    print("Output shape:", attention(x, cos, sin).shape)
