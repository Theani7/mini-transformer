import torch
from torch import nn

from .transformer_block import TransformerBlock


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

        self.max_seq_len = max_seq_len

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

        _, seq_len = tokens.shape

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

    # ponytail: no KV cache, re-runs the whole window each step (O(n^2));
    # add per-layer key/value caching if generation gets long.

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens,
        temperature=0.0
    ):

        for _ in range(max_new_tokens):

            window = tokens[:, -self.max_seq_len:]

            logits = self(window)[:, -1, :]

            if temperature == 0:
                next_token = logits.argmax(
                    dim=-1,
                    keepdim=True
                )
            else:
                probs = torch.softmax(
                    logits / temperature,
                    dim=-1
                )
                next_token = torch.multinomial(
                    probs,
                    num_samples=1
                )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

        return tokens


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