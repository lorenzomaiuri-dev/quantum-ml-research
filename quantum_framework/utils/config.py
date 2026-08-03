"""
Base configuration for Vision Transformer experiments (02 and 03).

BaseViTConfig holds all fields that are common to QViTConfig (experiment 02)
and AblationConfig (experiment 03). Each experiment extends it with its own
additional fields.

Usage:

    from quantum_framework.utils import BaseViTConfig
    from dataclasses import dataclass

    @dataclass
    class MyConfig(BaseViTConfig):
        my_extra_field: int = 42

Design notes:
- n_qubits is a computed property (= embed_dim). Experiments 02 and 03 both
  require embed_dim == n_qubits: data flows directly into AngleEmbedding
  without a classical bottleneck at the attention/embedding level.
- to_dict() includes all computed properties so that saved config.json files
  are fully self-describing and a run can be reproduced without re-running code.
"""

import torch
from dataclasses import dataclass, asdict


@dataclass
class BaseViTConfig:
    """
    Shared configuration for ViT-based quantum experiments.

    Fields common to experiments 02 (QViTConfig) and 03 (AblationConfig).
    Subclass this and add experiment-specific fields on top.
    """

    # --- Dataset ---
    dataset_name: str = (
        "pathmnist"  # MedMNIST key: pathmnist | bloodmnist | dermamnist | …
    )
    image_size: int = 28
    patch_size: int = 7
    n_channels: int = 3  # Updated automatically by the data loader
    n_classes: int = 9  # Updated automatically by the data loader

    # --- Transformer architecture ---
    embed_dim: int = 10  # Token embedding dimension; MUST equal n_qubits
    n_head: int = 2
    n_layer: int = 2
    ffn_dim: int = 40  # Feed-forward hidden dimension inside each block
    dropout: float = 0.1

    # --- Quantum circuit ---
    n_qlayers: int = 2  # Depth of StronglyEntanglingLayers
    q_device: str = "default.qubit"  # "lightning.gpu" selects NVIDIA/cuStateVec

    # --- Training ---
    seed: int = 42
    batch_size: int = 128
    max_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.embed_dim % self.n_head != 0:
            raise ValueError("embed_dim must be divisible by n_head")
        if min(self.embed_dim, self.n_head, self.n_layer, self.n_qlayers) <= 0:
            raise ValueError("model dimensions and layer counts must be positive")

    # --- Computed properties ---

    @property
    def n_patches(self) -> int:
        """Number of non-overlapping patches per image."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def seq_len(self) -> int:
        """Transformer sequence length = patches + 1 CLS token."""
        return self.n_patches + 1

    @property
    def patch_dim(self) -> int:
        """Flattened patch size in pixels × channels."""
        return self.patch_size * self.patch_size * self.n_channels

    @property
    def n_qubits(self) -> int:
        """
        Number of qubits in the VQC.

        Equals embed_dim by design. No classical bottleneck between the patch
        compression and the quantum circuit: each compressed feature maps
        directly to one qubit angle.
        """
        return self.embed_dim

    def to_dict(self) -> dict:
        """
        Serialize config to a JSON-serializable dict.

        Computed properties are included explicitly so that saved config.json
        files are fully self-describing without needing to re-instantiate
        the config object.
        """
        out = asdict(self)
        out["n_patches"] = self.n_patches
        out["seq_len"] = self.seq_len
        out["patch_dim"] = self.patch_dim
        out["n_qubits"] = self.n_qubits
        return out
