import torch
from torch import nn

from .feed_forward import SwiGLU
from .multi_head_attention import MultiHeadAttention
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = RMSNorm(d_model)
        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attention(self.norm1(x), cos, sin, cache)
        x = x + self.feed_forward(self.norm2(x))
        return x


if __name__ == "__main__":

    from .rope import rope_cache

    torch.manual_seed(42)

    x = torch.randn(1, 16, 32)
    cos, sin = rope_cache(16, 8)

    block = TransformerBlock(d_model=32, n_head=4, d_ff=64)

    print("Input shape:", x.shape)
    print("Output shape:", block(x, cos, sin).shape)
