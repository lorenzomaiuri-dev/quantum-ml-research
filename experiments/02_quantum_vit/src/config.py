"""
Configuration for experiment 02 — Quantum ViT on MedMNIST.

Extends BaseViTConfig with the fields specific to this experiment:
    use_quantum   — toggle between quantum and classical patch embedding
    eval_interval — evaluate every N epochs (useful when training is slow)

The embed_dim default is 8 here (vs 10 in experiment 03) because experiment 02
was the earlier prototype and used a smaller embedding to keep simulation fast.
"""

from dataclasses import dataclass
from quantum_framework.utils import BaseViTConfig


@dataclass
class QViTConfig(BaseViTConfig):
    """Configuration for the Quantum ViT experiment (02)."""

    # Override defaults that differ from BaseViTConfig
    dataset_name: str = "pathmnist"
    embed_dim: int = 8  # Smaller than experiment 03; must equal n_qubits
    ffn_dim: int = 32
    batch_size: int = 128
    max_epochs: int = 50
    train_subset: int = 0

    # --- Experiment-02-specific ---
    use_quantum: bool = True  # False → classical linear patch embedding for baseline
    eval_interval: int = 1  # Evaluate every epoch
