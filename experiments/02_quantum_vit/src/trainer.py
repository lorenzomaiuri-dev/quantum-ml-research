"""
Trainer for experiment 02 — Quantum ViT on MedMNIST.

Thin wrapper around quantum_framework.training.BaseTrainer.
Handles experiment-specific setup: model instantiation, data loading,
run directory creation.

This is the trainer used by experiments/02_quantum_vit/main.py.
The older src/engine/trainer.py (HybridQCNNViT variant) is superseded
by this file for new runs but kept for reference.
"""

from quantum_framework.training import BaseTrainer
from quantum_framework.utils import make_run_dir, set_seed

from src.dataset import load_dataset
from src.model import QuantumViT


class Trainer(BaseTrainer):
    """
    Trainer for the Quantum ViT experiment (02).

    Args:
        config:          QViTConfig instance.
        experiment_name: Optional label for the run directory.
    """

    def __init__(self, config, seed=42, experiment_name=None):
        set_seed(seed)
        config.seed = seed
        # Data (also writes n_channels, n_classes back onto config)
        train_loader, val_loader, test_loader = load_dataset(config)

        # Model
        model = QuantumViT(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: {'Quantum' if config.use_quantum else 'Classical'} ViT")
        print(f"  Total params:     {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")

        tag = experiment_name or ("quantum" if config.use_quantum else "classical")
        run_dir = make_run_dir("experiments", f"qvit_{tag}")

        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_dir=run_dir,
            seed=seed,
            # Experiment 02 does not use diagnostic hooks.
            log_gradients=False,
            log_activations=False,
            grad_clip=1.0,
            experiment_id="02_quantum_vit_attention",
            variant=tag,
            evaluate_noise=False,
        )
