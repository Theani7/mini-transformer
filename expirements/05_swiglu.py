import torch

from mini_transformer.feed_forward import SwiGLU

torch.manual_seed(42)

x = torch.randn(1, 4, 16)

ffn = SwiGLU(d_model=16, d_ff=32)

print("gate(x):", ffn.gate(x).shape)
print("silu(gate(x)) * up(x):", (torch.nn.functional.silu(ffn.gate(x)) * ffn.up(x)).shape)
print("output:", ffn(x).shape)
