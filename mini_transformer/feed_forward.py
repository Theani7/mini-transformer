import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """Superseded by SwiGLU; removed in Task 6."""

    def __init__(self, d_model, d_ff):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.network(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


if __name__ == "__main__":

    torch.manual_seed(42)

    x = torch.randn(1, 4, 16)
    ffn = SwiGLU(d_model=16, d_ff=32)

    print("Input shape:", x.shape)
    print("Output shape:", ffn(x).shape)
