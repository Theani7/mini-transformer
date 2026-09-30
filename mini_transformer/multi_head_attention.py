import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Split into heads
        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        # Move heads before sequence dimension
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention scores
        scores = Q @ K.transpose(-2, -1)

        # Scale
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len)
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Attention weights
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # Weighted values
        attention_output = attention_weights @ V

        # Move sequence before heads
        attention_output = attention_output.transpose(1, 2)

        # Combine heads
        attention_output = attention_output.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

        # Final projection
        output = self.out_projection(attention_output)

        return output


# Experiment

if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(1, 4, 8)

    attention = MultiHeadAttention(
        d_model=8,
        num_heads=2
    )

    output = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)