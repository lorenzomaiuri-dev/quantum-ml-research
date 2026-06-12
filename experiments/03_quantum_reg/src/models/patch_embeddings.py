"""
Three Patch Embedding variants for the ablation study.

All three share the same classical compression layer Linear(patch_dim, embed_dim).
They differ ONLY in what happens after compression:

    A. VanillaPatchEmbedding     — nothing, raw linear output
    B. BoundedMLPPatchEmbedding  — tanh·π → MLP(10→20→10) → tanh
    C. QuantumRegPatchEmbedding  — tanh·π → AngleEmbedding → StronglyEntangling → ⟨Z⟩

The compression layer is the same nn.Linear in all three, so gradient
differences are attributable to the post-processing path only.
"""

import torch
import torch.nn as nn

from quantum_framework.layers import QuantumLinear


class VanillaPatchEmbedding(nn.Module):
    """
    Model A: Standard linear patch embedding.
    Patch → Linear(patch_dim, embed_dim)
    No constraints, no bounding. Baseline for overfitting measurement.

    Params: patch_dim × embed_dim (e.g. 147 × 10 = 1,470)
    """

    def __init__(self, config):
        super().__init__()
        self.compression = nn.Linear(config.patch_dim, config.embed_dim)

    def forward(self, x):
        # x: (B, n_patches, patch_dim)
        return self.compression(x)


class BoundedMLPPatchEmbedding(nn.Module):
    """
    Model B: Bounded classical MLP — the strong rival.
    Patch → Linear(patch_dim, embed_dim) → tanh·π → MLP(embed→hidden→embed) → tanh

    This mimics the bounded I/O of the quantum path WITHOUT unitarity or
    entanglement. The MLP has intentionally MORE parameters than the VQC
    (e.g. 400 vs 60) to make any quantum advantage more convincing.

    Params: patch_dim × embed_dim + embed_dim × hidden + hidden × embed_dim
            e.g. 1,470 + 10×20 + 20×10 = 1,870
    """

    def __init__(self, config):
        super().__init__()
        self.compression = nn.Linear(config.patch_dim, config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_hidden, bias=False),
            nn.GELU(),
            nn.Linear(config.mlp_hidden, config.embed_dim, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: (B, n_patches, patch_dim)
        x = self.compression(x)
        x = torch.tanh(x) * torch.pi  # Bounded to [-π, π]
        x = self.mlp(x)  # Output bounded to [-1, 1] by final tanh
        return x


class QuantumRegPatchEmbedding(nn.Module):
    """
    Model C: Quantum-Regularized Patch Embedding.
    Patch → Linear(patch_dim, embed_dim) → tanh·π → AngleEmbedding → StronglyEntangling → ⟨Z⟩

    Same compression and bounding as Model B. The VQC adds:
    - Unitarity: gates preserve state vector norms
    - Entanglement: correlations between qubits create non-separable features

    The VQC is implemented by QuantumLinear from quantum_framework.layers,
    which handles the tanh·π scaling and angle embedding internally.

    Params: patch_dim × embed_dim + n_qlayers × n_qubits × 3
            e.g. 1,470 + 2 × 10 × 3 = 1,530
    """

    def __init__(self, config):
        super().__init__()
        # embed_dim == n_qubits by design (enforced by AblationConfig.n_qubits property)
        self.compression = nn.Linear(config.patch_dim, config.embed_dim)
        self.vqc = QuantumLinear(config.n_qubits, config.n_qlayers, config.q_device)

    def forward(self, x):
        # x: (B, n_patches, patch_dim)
        x = self.compression(x)
        # QuantumLinear accepts (..., n_qubits) and handles tanh·π + circuit internally.
        return self.vqc(x)  # (B, n_patches, n_qubits)


def build_patch_embedding(config, model_type: str) -> nn.Module:
    """Factory function to create the correct patch embedding."""
    builders = {
        "vanilla": VanillaPatchEmbedding,
        "bounded_mlp": BoundedMLPPatchEmbedding,
        "quantum_reg": QuantumRegPatchEmbedding,
    }
    if model_type not in builders:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(builders.keys())}")
    return builders[model_type](config)
