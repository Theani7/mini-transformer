import torch
import torch.nn as nn

vocab_size =8
embedding_dim = 4

embedding = nn.Embedding(vocab_size,embedding_dim)

tokens = torch.tensor([3,2,4,4,5])

vectors = embedding(tokens)

print("Tokens:")
print(tokens)

print("\n Embeddings")
print(embedding)

print("\n Shape:")
print(vectors.shape)