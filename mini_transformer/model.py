import torch
from torch import nn

from .config import Config
from .rmsnorm import RMSNorm
from .rope import rope_cache
from .sampling import sample_next
from .transformer_block import TransformerBlock


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, config.d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=config.d_model,
                    n_head=config.n_head,
                    d_ff=config.d_ff,
                )
                for _ in range(config.n_layer)
            ]
        )

        self.norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens, cos, sin):
        x = self.token_embedding(tokens)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, config=None, temperature=1.0, top_p=1.0):
        config = config or self.config
        cos, sin = rope_cache(config.block_size, config.head_dim, device=tokens.device)

        for _ in range(max_new_tokens):
            window = tokens[:, -config.block_size :]
            logits = self(window, cos, sin)[:, -1, :]
            next_token = sample_next(logits, temperature=temperature, top_p=top_p)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens


if __name__ == "__main__":

    config = Config()

    torch.manual_seed(42)

    model = MiniTransformer(vocab_size=128, config=config)
    cos, sin = rope_cache(config.block_size, config.head_dim)

    tokens = torch.randint(0, 128, (1, 8))

    print("Input shape:", tokens.shape)
    print("Logits shape:", model(tokens, cos, sin).shape)
