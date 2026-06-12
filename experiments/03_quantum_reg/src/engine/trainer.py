"""
Trainer for experiment 03 — Quantum Regularization Ablation Study.

Thin wrapper around quantum_framework.training.BaseTrainer.
Handles experiment-specific setup: model instantiation, data loading,
run directory creation, and hook wiring.
"""

from quantum_framework.training import BaseTrainer
from quantum_framework.utils import make_run_dir, set_seed

from src.data.loaders import load_dataset
from src.models.vit import AblationViT


class Trainer(BaseTrainer):
    """
    Trainer for the ablation study (experiment 03).

    Instantiates AblationViT and loads MedMNIST data, then delegates all
    training logic to BaseTrainer.

    Args:
        config:     AblationConfig instance.
        model_type: "vanilla" | "bounded_mlp" | "quantum_reg"
        seed:       Random seed for reproducibility.
        run_tag:    Optional label override for the run directory.
    """

    def __init__(self, config, model_type: str, seed: int, run_tag: str = ""):
        self.model_type = model_type
        self.seed = seed

        # Data
        train_loader, val_loader, test_loader = load_dataset(config)

        # Model
        model = AblationViT(config, model_type)
        param_counts = model.count_params()
        print(f"  Model: {model_type} | Seed: {seed}")
        print(
            f"  Params — total: {param_counts['total']:,} | "
            f"patch_embed: {param_counts['patch_embed']:,} | "
            f"transformer: {param_counts['transformer']:,}"
        )

        tag = run_tag or f"{model_type}_{config.dataset_name}_s{seed}"
        run_dir = make_run_dir("experiments", tag)

        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_dir=run_dir,
            seed=seed,
            log_gradients=config.log_gradients,
            log_activations=config.log_activations,
            monitor_module_name="patch_embed.compression",
            grad_clip=1.0,
        )

    def train(self):
        print(f"\n{'='*60}")
        print(f"  Training {self.model_type} | {self.config.dataset_name} | seed={self.seed}")
        run_dir, results = super().train()
        # Annotate results with experiment-03-specific fields.
        results["model_type"] = self.model_type
        results["dataset"]    = self.config.dataset_name
        results["seed"]       = self.seed
        return run_dir, results
