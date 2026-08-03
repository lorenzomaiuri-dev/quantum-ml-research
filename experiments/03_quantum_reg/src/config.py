"""
Configuration for experiment 03 — Quantum Regularization Ablation Study.

Extends BaseViTConfig with the fields specific to this experiment:
    mlp_hidden    — hidden dim of the MLP in Model B (BoundedMLPPatchEmbedding)
    train_subset  — stratified subset size (0 = full dataset; >0 = scarce-data regime)
    seeds         — list of seeds for the full ablation (3 models × 5 seeds × 3 datasets)
    datasets      — list of MedMNIST dataset names to sweep over
    log_gradients — record L2 norm of compression-layer gradients each epoch
    log_activations — record activation distribution stats each epoch

The key metric is generalization_gap = train_acc − test_acc.
A valid quantum regularization result requires the gap for quantum_reg to be
lower than bounded_mlp across 3+ datasets and 5+ seeds with p < 0.05.
"""

from dataclasses import dataclass, field
from typing import List
from quantum_framework.utils import BaseViTConfig


@dataclass
class AblationConfig(BaseViTConfig):
    """
    Configuration for the Quantum Regularization Ablation Study (experiment 03).

    Three models are tested:
        A. Vanilla ViT        — Linear(patch_dim, embed_dim), no constraints
        B. Bounded MLP        — Linear + tanh·π + MLP(10→20→10) + tanh
        C. Quantum-Regularized — Linear + tanh·π + VQC(AngleEmbed → StronglyEntangling → ⟨Z⟩)

    The key comparison is B vs C: same bounded I/O, same compression layer,
    but C adds unitarity and entanglement via the VQC. If C shows lower
    generalization gap, the effect is attributable to quantum geometry.
    """

    # Override defaults that differ from BaseViTConfig
    embed_dim: int = 10   # Must equal n_qubits (enforced by BaseViTConfig.n_qubits property)
    ffn_dim: int = 40

    # --- Model B specific ---
    mlp_hidden: int = 20  # Hidden dim of BoundedMLPPatchEmbedding; intentionally > VQC params

    # --- Data regime ---
    train_subset: int = 0  # 0 = full dataset. >0 = stratified subset of N samples
    eval_interval: int = 5  # Full validation cadence; final epoch is always evaluated

    # --- Ablation sweep ---
    seeds: List[int] = field(default_factory=lambda: [42, 137, 256, 512, 1024])
    datasets: List[str] = field(
        default_factory=lambda: ["pathmnist", "bloodmnist", "dermamnist"]
    )

    # --- Diagnostic logging ---
    log_gradients: bool = True    # Log L2 norm of compression layer gradients
    log_activations: bool = True  # Log activation distribution stats (saturation detection)
