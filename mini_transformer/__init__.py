"""Mini-transformer: A decoder-only Transformer language model from scratch in PyTorch."""

from .config import Config
from .model import MiniTransformer
from .tokenizer import decode, encode, load_tokenizer

__version__ = "0.2.0"

__all__ = [
    "Config",
    "MiniTransformer",
    "__version__",
    "decode",
    "encode",
    "load_tokenizer",
]
