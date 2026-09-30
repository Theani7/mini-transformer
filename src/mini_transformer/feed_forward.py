import torch
from torch import nn


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(1, 4, 8)

    ffn = FeedForward(
        d_model=8,
        d_ff=32
    )

    output = ffn(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)