import torch

from mini_transformer.rope import apply_rope, rope_cache, rotate_half

torch.manual_seed(42)

head_dim = 8
cos, sin = rope_cache(8, head_dim)

q = torch.randn(1, 1, 1, head_dim)

print("cos[0, 0]:", cos[0, 0])
print("rotate_half(q):", rotate_half(q))
print("rotated q at position 3:", apply_rope(q, cos[:, :, 3:4], sin[:, :, 3:4]))
