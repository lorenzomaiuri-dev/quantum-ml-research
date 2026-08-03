"""Pre-registered configuration for the definitive thesis campaign."""

from dataclasses import dataclass

from .base import GPTConfig as BaseConfig


@dataclass
class GPTConfig(BaseConfig):
    tokenizer_class: str = "CharTokenizer"
    batch_size: int = 8
    block_size: int = 16
    max_iters: int = 1500
    eval_interval: int = 100
    eval_iters: int = 100
    learning_rate: float = 3e-3
    n_embd: int = 8
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    use_quantum: bool = False
