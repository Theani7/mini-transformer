import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff
        )

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(
            self.norm1(x)
        )

        # Feed-forward + residual
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(1, 4, 8)

    block = TransformerBlock(
        d_model=8,
        num_heads=2,
        d_ff=32
    )

    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)