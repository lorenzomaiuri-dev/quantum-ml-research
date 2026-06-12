import torch
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    tokenizer_class: str = "BiCharTokenizer"
    batch_size: int = 32
    block_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 100
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters: int = 200
    n_embd: int = 32
    n_head: int = 8
    n_layer: int = 4
    dropout: float = 0.05
    use_quantum: bool = False
    n_qlayers: int = 2
    q_device: str = "default.qubit"

    @property
    def n_qubits(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        out = asdict(self)
        out["n_qubits"] = self.n_qubits  # computed property not included by asdict
        return out
