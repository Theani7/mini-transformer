import json
import sys
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


def resolve_checkpoint_config(saved, requested=None, path="", ignore=()):
    """Reconcile a checkpoint's stored config with a caller's, and return the
    config to load it under. Raises `ValueError` naming what disagrees, because
    that is the one thing both CLIs already turn into a one-line message.

    `saved` is the checkpoint's `config` entry, or None if it stored none.
    `requested` is the caller's own config; None means the checkpoint's wins.
    `ignore` names fields the caller is allowed to change on purpose - `--resume`
    exists to change `iters`, so a diff there would forbid extending a run.

    Both checkpoint loaders go through here, so the "two loaders, two contracts"
    split - a mismatched checkpoint refused by `generate` and silently resumed by
    `train` - cannot come back.
    """
    # Checked before *both* uses of `saved` below. Filtering only at the
    # `Config(**saved)` call would turn this into an AttributeError further down,
    # because the diff loop still walks the raw dict.
    unknown = set(saved or {}) - Config.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(
            f"{path} was written by a different version "
            f"({', '.join(sorted(unknown))}) - retrain, or check out the "
            f"version that wrote it"
        )

    if saved is None:
        print(
            f"warning: {path} stored no config; loading it under the current "
            f"defaults, which may not be the hyperparameters it was trained with",
            file=sys.stderr,
        )
        return Config() if requested is None else requested

    if requested is None:
        return Config(**saved)

    differing = {
        key: (saved[key], getattr(requested, key))
        for key in saved
        if key not in ignore and saved[key] != getattr(requested, key)
    }
    if differing:
        named = ", ".join(
            f"{k} (checkpoint {old!r}, requested {new!r})"
            for k, (old, new) in sorted(differing.items())
        )
        raise ValueError(f"checkpoint config differs: {named}")

    return requested
