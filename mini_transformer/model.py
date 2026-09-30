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

    def forward(self, tokens, cos, sin, cache=None):
        seq_len = tokens.shape[1]

        # RoPE reads absolute position, so a cached decode step at position
        # `past` must slice the table there, not at zero.
        past = 0
        if cache is not None and cache[0].get("k") is not None:
            past = cache[0]["k"].shape[2]

        cos = cos[:, :, past : past + seq_len]
        sin = sin[:, :, past : past + seq_len]

        x = self.token_embedding(tokens)

        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if cache is None else cache[i])

        x = self.norm(x)

        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens,
        config=None,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        use_cache=False,
        on_token=None,
    ):
        # Measured on the 3.03M model: 1.68 ms/token uncached vs 1.85 cached over
        # 300 tokens on MPS. Decode at this scale is dominated by fixed per-step
        # overhead, not by the arithmetic the cache removes, so it is opt-in. It
        # pays off once decode is compute-bound — a larger model, or CUDA with
        # graph capture, where per-step launch cost stops dominating.
        config = config or self.config
        # Cached keys carry their absolute rotation, so positions stay contiguous
        # and correct past block_size; RoPE extrapolates by construction.
        cos, sin = rope_cache(
            config.block_size + max_new_tokens,
            config.head_dim,
            device=tokens.device,
        )

        cache = [{} for _ in self.blocks] if use_cache else None
        window = tokens[:, -config.block_size :]

        for _ in range(max_new_tokens):
            logits = self(window, cos, sin, cache)[:, -1, :]
            next_token = sample_next(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                context=tokens,
            )
            tokens = torch.cat([tokens, next_token], dim=1)
            window = next_token if use_cache else tokens[:, -config.block_size :]
            if on_token is not None:
                on_token(next_token)

        return tokens


if __name__ == "__main__":

    config = Config()

    torch.manual_seed(42)

    model = MiniTransformer(vocab_size=128, config=config)
    cos, sin = rope_cache(config.block_size, config.head_dim)

    tokens = torch.randint(0, 128, (1, 8))

    print("Input shape:", tokens.shape)
    print("Logits shape:", model(tokens, cos, sin).shape)
