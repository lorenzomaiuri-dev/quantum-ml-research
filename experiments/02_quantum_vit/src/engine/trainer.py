"""
Trainer for experiment 02 — Quantum ViT on MedMNIST (engine variant).

Thin wrapper around quantum_framework.training.BaseTrainer.
Uses HybridQCNNViT (quantum patch embedding via QCNN) and routes data
through the base/torchvision dispatcher so that non-MedMNIST datasets
(cifar10, fashionmnist) continue to work.
"""

from quantum_framework.training import BaseTrainer
from quantum_framework.utils import make_run_dir, set_seed

from src.data.base import get_dataloaders
from src.models.hybrid_vit import HybridQCNNViT


class Trainer(BaseTrainer):
    """
    Trainer for the Quantum ViT experiment (02), engine variant.

    Uses HybridQCNNViT, which applies a quantum patch embedding (QCNN)
    followed by a classical TransformerEncoder. Toggle use_quantum=False
    in QViTConfig for the classical baseline.

    Args:
        config:          QViTConfig instance.
        experiment_name: Optional label for the run directory.
    """

    def __init__(self, config, seed=42, experiment_name=None, run_dir=None):
        # Seed before dataset shuffling and model parameter initialisation.
        set_seed(seed)
        config.seed = seed
        # Data — routes to MedMNIST or torchvision depending on dataset_name
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # Model
        model = HybridQCNNViT(config)
        tag = "quantum" if config.use_quantum else "classical"
        name = experiment_name or f"hybrid_{tag}_{config.dataset_name}"
        run_dir = run_dir or make_run_dir("experiments", name)

        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_dir=run_dir,
            seed=seed,
            log_gradients=False,
            log_activations=False,
            grad_clip=1.0,
            experiment_id="02_quantum_vit",
            variant=tag,
            evaluate_noise=False,
        )
