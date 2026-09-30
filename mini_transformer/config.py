import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # data
    dataset: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    corpus_chars: int = 200_000
    bpe_vocab_size: int = 8192
    # model
    d_model: int = 192
    n_layer: int = 4
    n_head: int = 6
    d_ff: int = 512
    block_size: int = 128
    # optimisation
    batch_size: int = 8
    iters: int = 6000
    lr: float = 6e-3
    warmup: int = 100
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # bookkeeping
    seed: int = 42
    eval_interval: int = 500
    sample_interval: int = 1000

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    @property
    def min_lr(self):
        return self.lr * self.min_lr_ratio

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
