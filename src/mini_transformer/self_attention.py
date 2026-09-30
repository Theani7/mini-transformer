import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 3 tokens
# 4-dimensional representations

X = torch.rand(3, 4)


# Learnable projection matrices
W_Q = torch.rand(4, 4)
W_K = torch.rand(4, 4)
W_V = torch.rand(4, 4)


# Create Query, Key, Value
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("X shape:", X.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)


# Calculate attention scores
scores = Q @ K.T

print("\nScores:")
print(scores)


# Scale
d_k = K.shape[-1]

scaled_scores = scores / (d_k ** 0.5)

print("\nScaled scores:")
print(scaled_scores)


# Create causal mask
mask = torch.tril(
    torch.ones(X.shape[0], X.shape[0])
)

print("\nMask:")
print(mask)


# Apply causal mask
masked_scores = scaled_scores.masked_fill(
    mask == 0,
    float("-inf")
)

print("\nMasked scores:")
print(masked_scores)


# Convert scores into attention weights
attention_weights = F.softmax(
    masked_scores,
    dim=-1
)

print("\nAttention weights:")
print(attention_weights)


# Weighted combination of values
output = attention_weights @ V

print("\nOutput:")
print(output)

print("\nOutput shape:", output.shape)