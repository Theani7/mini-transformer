"""Model architecture summary and parameter inspector for MiniTransformer."""

import argparse
from pathlib import Path

from .config import Config
from .generate import load_checkpoint
from .model import MiniTransformer
from .tokenizer import load_tokenizer


def model_summary(model, config, vocab_size):
    """Generate a structured dictionary summarizing model architecture and memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.token_embedding.parameters())
    # lm_head is tied to token_embedding, so unique non-embedding params are:
    stack_params = total_params - embed_params

    # Memory in bytes (FP32 = 4 bytes, BF16/FP16 = 2 bytes)
    fp32_mb = (total_params * 4) / (1024 * 1024)
    bf16_mb = (total_params * 2) / (1024 * 1024)

    # Theoretical FLOPs per forward pass token ~= 2 * total_params
    flops_per_token = 2 * total_params

    layers = []
    layers.append(
        {
            "component": "Token Embedding",
            "shape": f"({vocab_size}, {config.d_model})",
            "params": embed_params,
            "detail": "tied to lm_head",
        }
    )
    for i, block in enumerate(model.blocks):
        attn_params = sum(p.numel() for p in block.attention.parameters())
        ffn_params = sum(p.numel() for p in block.feed_forward.parameters())
        norm_params = sum(p.numel() for p in block.norm1.parameters()) + sum(
            p.numel() for p in block.norm2.parameters()
        )
        layers.append(
            {
                "component": f"TransformerBlock {i}",
                "shape": f"d_model={config.d_model}, n_head={config.n_head}, d_ff={config.d_ff}",
                "params": attn_params + ffn_params + norm_params,
                "detail": f"MHA(RoPE)={attn_params:,} | SwiGLU={ffn_params:,} | RMSNorms={norm_params:,}",
            }
        )
    final_norm_params = sum(p.numel() for p in model.norm.parameters())
    layers.append(
        {
            "component": "Final RMSNorm",
            "shape": f"({config.d_model},)",
            "params": final_norm_params,
            "detail": "pre-head normalization",
        }
    )
    layers.append(
        {
            "component": "LM Head (tied)",
            "shape": f"({vocab_size}, {config.d_model})",
            "params": 0,
            "detail": "shared with token_embedding",
        }
    )

    return {
        "total_params": total_params,
        "embed_params": embed_params,
        "stack_params": stack_params,
        "fp32_mb": fp32_mb,
        "bf16_mb": bf16_mb,
        "flops_per_token": flops_per_token,
        "config": config,
        "vocab_size": vocab_size,
        "layers": layers,
    }


def print_summary(summary):
    c = summary["config"]
    print("=" * 70)
    print(" MiniTransformer Architecture Summary")
    print("=" * 70)
    print(f" Vocabulary:     {summary['vocab_size']:,} tokens")
    print(f" Context Window: {c.block_size} tokens")
    print(
        f" Architecture:   {c.n_layer} layers | {c.d_model} d_model | {c.n_head} heads | {c.d_ff} d_ff"
    )
    print(f" Attention:      RoPE (head_dim={c.head_dim}) | RMSNorm | SwiGLU | Tied Head")
    print("-" * 70)
    print(f" {'Layer':<24} {'Specs / Shape':<24} {'Parameters':>18}")
    print("-" * 70)
    for layer in summary["layers"]:
        p_str = f"{layer['params']:,}" if layer["params"] > 0 else "(tied)"
        print(f" {layer['component']:<24} {layer['shape']:<24} {p_str:>18}")
    print("-" * 70)
    print(f" Total Parameters:     {summary['total_params']:,}")
    print(f" Transformer Stack:    {summary['stack_params']:,}")
    print(f" Embedding (tied):     {summary['embed_params']:,}")
    print(f" FP32 Memory:          {summary['fp32_mb']:.2f} MB")
    print(f" BF16 / FP16 Memory:   {summary['bf16_mb']:.2f} MB")
    print(f" FLOPs / token (fwd):  ~{summary['flops_per_token']:,}")
    print("=" * 70)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Inspect MiniTransformer architecture and parameters"
    )
    p.add_argument(
        "--checkpoint", default=None, help="path to .pt or .safetensors checkpoint"
    )
    p.add_argument(
        "--tokenizer", default="data/tokenizer.json", help="path to tokenizer.json"
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if args.checkpoint and Path(args.checkpoint).exists():
        tok = load_tokenizer(args.tokenizer) if Path(args.tokenizer).exists() else None
        vocab_size = tok.get_vocab_size() if tok else 8192
        model, config = load_checkpoint(args.checkpoint, args.device, vocab_size)
    else:
        config = Config()
        tok = load_tokenizer(args.tokenizer) if Path(args.tokenizer).exists() else None
        vocab_size = tok.get_vocab_size() if tok else config.bpe_vocab_size
        model = MiniTransformer(vocab_size=vocab_size, config=config)

    summary = model_summary(model, config, vocab_size)
    print_summary(summary)


if __name__ == "__main__":
    main()
