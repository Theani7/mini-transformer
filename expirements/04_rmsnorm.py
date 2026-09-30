import torch

from mini_transformer.rmsnorm import RMSNorm

torch.manual_seed(42)

x = torch.randn(1, 4, 16) * 5

norm = RMSNorm(16)
out = norm(x)

print("input rms :", x.pow(2).mean().sqrt().item())
print("output rms:", out.pow(2).mean().sqrt().item())
