"""Mini-transformer: A decoder-only Transformer language model from scratch in PyTorch."""

from .config import Config
from .model import MiniTransformer
from .tokenizer import (
    EOT,
    IM_END,
    IM_START,
    decode,
    encode,
    format_chat,
    load_tokenizer,
)

__version__ = "0.2.0"

__all__ = [
    "EOT",
    "IM_END",
    "IM_START",
    "Config",
    "MiniTransformer",
    "__version__",
    "decode",
    "encode",
    "format_chat",
    "load_tokenizer",
]
