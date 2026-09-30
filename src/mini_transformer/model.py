import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff):
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

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class MiniTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_seq_len
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, tokens):

        batch_size, seq_len = tokens.shape

        positions = torch.arange(
            seq_len,
            device=tokens.device
        )

        token_embeddings = self.token_embedding(tokens)

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


if __name__ == "__main__":

    torch.manual_seed(42)

    model = MiniTransformer(
        vocab_size=8,
        d_model=32,
        num_heads=4,
        d_ff=128,
        num_layers=2,
        max_seq_len=64
    )

    tokens = torch.tensor([
        [3, 2, 4, 4, 5]
    ])

    logits = model(tokens)

    print("Input shape:", tokens.shape)
    print("Logits shape:", logits.shape)