"""
Generic image-classification training loop for hybrid quantum-classical ViT experiments.

BaseTrainer encapsulates the training infrastructure that is identical between
experiments 02 and 03:

    - AdamW optimiser + CosineAnnealing LR scheduler
    - Per-epoch tqdm progress bar
    - TensorBoard SummaryWriter
    - Best-checkpoint saving (by val_acc)
    - Optional gradient-norm monitoring on a named module (experiment 03)
    - Optional activation-saturation monitoring on a named module (experiment 03)
    - Noise robustness evaluation at the end of training (experiment 03)
    - Standardised output files: config.json, params.json, best_model.pth,
      final_model.pth, results.json

Usage:

    from quantum_framework.training import BaseTrainer

    class MyTrainer(BaseTrainer):
        def __init__(self, config, model_type, seed):
            model = MyModel(config, model_type)
            train_loader, val_loader, test_loader = load_my_data(config)
            run_dir = make_run_dir("experiments", f"{model_type}_s{seed}")
            super().__init__(
                config, model,
                train_loader, val_loader, test_loader,
                run_dir,
                seed=seed,
                log_gradients=config.log_gradients,
                log_activations=config.log_activations,
                monitor_module_name="patch_embed.compression",
            )

    trainer = MyTrainer(config, "quantum_reg", seed=42)
    run_dir, results = trainer.train()

The model must follow the (images, labels) → (logits, loss) convention used
by AblationViT (exp 03) and QuantumViT (exp 02). Passing labels=None makes
the model return loss=None; evaluate() always passes labels.

Diagnostic hooks:
    When log_gradients=True, a backward hook on `monitor_module_name` records
    the L2 norm of the output gradient each step. Useful for detecting gradient
    vanishing through tanh-bounded layers.

    When log_activations=True, a forward hook on the same module records mean,
    std, abs_max, and saturation_frac (fraction for which
    |tanh(pre-activation)| > 0.95).
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from quantum_framework.data import make_noisy_loader
from quantum_framework.evaluation import (
    compute_classification_metrics,
    generalization_gap,
)
from quantum_framework.utils import collect_run_metadata, set_seed, save_json


class BaseTrainer:
    """
    Generic training loop for image-classification experiments.

    Args:
        config:               Config dataclass (must have .device, .max_epochs,
                              .learning_rate, .weight_decay, and .to_dict()).
        model:                nn.Module following the (images, labels?) → (logits, loss?) API.
        train_loader:         DataLoader for the training split.
        val_loader:           DataLoader for the validation split.
        test_loader:          DataLoader for the test split.
        run_dir:              Pre-created directory for logs and checkpoints.
        seed:                 Random seed (set on construction for reproducibility).
        log_gradients:        If True, record L2 norm of gradients on monitor_module_name.
        log_activations:      If True, record activation statistics on monitor_module_name.
        monitor_module_name:  Dot-path to the submodule to hook (e.g. "patch_embed.compression").
                              Required when log_gradients or log_activations is True.
        grad_clip:            Max gradient norm for clipping (0.0 = no clipping).
    """

    def __init__(
        self,
        config,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        run_dir: str,
        seed: int = 42,
        log_gradients: bool = False,
        log_activations: bool = False,
        monitor_module_name: str = "",
        grad_clip: float = 1.0,
        experiment_id: str = "",
        variant: str = "",
        evaluate_noise: bool = False,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.run_dir = run_dir
        self.log_gradients = log_gradients
        self.log_activations = log_activations
        self.monitor_module_name = monitor_module_name
        self.grad_clip = grad_clip
        self.seed = seed
        self.experiment_id = experiment_id
        self.variant = variant
        self.evaluate_noise = evaluate_noise

        set_seed(seed)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_epochs
        )
        self.writer = SummaryWriter(log_dir=run_dir)

        # Storage for hook data, reset each epoch.
        self._grad_norms: list = []
        self._activation_stats: list = []

    # ------------------------------------------------------------------
    # Diagnostic hooks
    # ------------------------------------------------------------------

    def _parameter_counts(self) -> dict:
        if hasattr(self.model, "count_params"):
            return self.model.count_params()
        return {
            "total": sum(p.numel() for p in self.model.parameters()),
            "trainable": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }

    def _resolve_module(self, name: str) -> nn.Module:
        """Traverse dot-path to find a submodule (e.g. 'patch_embed.compression')."""
        module = self.model
        for part in name.split("."):
            module = getattr(module, part)
        return module

    def _register_hooks(self) -> None:
        """
        Register backward and forward hooks on monitor_module_name.

        Backward hook: records the L2 norm of the gradient flowing out of
        the module. A shrinking norm over training suggests gradient filtering
        by tanh saturation.

        Forward hook: records mean, std, abs_max, and the fraction of output
        values that are near tanh saturation (|x| > 0.95·π).
        """
        module = self._resolve_module(self.monitor_module_name)

        def grad_hook(mod, grad_input, grad_output):
            if grad_output[0] is not None:
                self._grad_norms.append(grad_output[0].detach().norm(2).item())

        def act_hook(mod, inp, output):
            with torch.no_grad():
                out = output.detach()
                self._activation_stats.append(
                    {
                        "mean": out.mean().item(),
                        "std": out.std().item(),
                        "abs_max": out.abs().max().item(),
                        # The hook observes the raw compression output, before tanh.
                        # |tanh(x)| > 0.95 iff |x| > arctanh(0.95) ≈ 1.83.
                        "saturation_frac": (out.abs() > math.atanh(0.95))
                        .float()
                        .mean()
                        .item(),
                    }
                )

        module.register_full_backward_hook(grad_hook)
        module.register_forward_hook(act_hook)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        """
        Compute loss, accuracy, AUC, and F1 on a given DataLoader.

        The model is set to eval mode for the duration and restored to train mode
        before returning (so this can safely be called mid-epoch).

        Returns:
            dict with keys: loss, accuracy, auc, f1
        """
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss, n_samples = 0.0, 0

        for images, labels in loader:
            images = images.to(self.config.device)
            labels = labels.to(self.config.device)

            logits, loss = self.model(images, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        self.model.train()

        metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / max(n_samples, 1)
        return metrics

    # ------------------------------------------------------------------
    # Noise robustness
    # ------------------------------------------------------------------

    def _test_noise_robustness(self) -> dict:
        """
        Evaluate the trained model under Gaussian and salt-and-pepper noise.

        Returns nested loss, accuracy, AUC and F1 values for every noise type
        and intensity.

        Called automatically at the end of train() when the test_loader is
        a MedMNIST-based loader (experiment 03). Override or skip in subclasses
        where noise robustness is not part of the evaluation protocol.
        """
        print("  Testing noise robustness...")
        noise_types = ["gaussian", "salt_pepper"]
        intensities = [0.05, 0.1, 0.2, 0.3, 0.5]
        results = {}

        for noise_index, noise_type in enumerate(noise_types):
            results[noise_type] = {}
            for intensity_index, intensity in enumerate(intensities):
                # Identical corruptions for variants sharing the same run seed.
                set_seed(self.seed + 10_000 + noise_index * 100 + intensity_index)
                noisy_loader = make_noisy_loader(
                    self.test_loader, noise_type, intensity, self.config.batch_size
                )
                m = self.evaluate(noisy_loader)
                results[noise_type][str(intensity)] = {
                    key: round(m[key], 4) for key in ("loss", "accuracy", "auc", "f1")
                }

            accs = [entry["accuracy"] for entry in results[noise_type].values()]
            print(f"    {noise_type}: {accs}")

        return results

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> tuple:
        """
        Run the full training loop and return (run_dir, results).

        Training sequence:
            1. Save config.json and params.json to run_dir.
            2. Register diagnostic hooks if requested.
            3. For each epoch: forward pass, backward pass, scheduler step,
               val evaluation, TensorBoard logging, best-checkpoint save.
            4. Load best checkpoint and evaluate on test set.
            5. Run noise robustness test.
            6. Save results.json and final_model.pth.

        Returns:
            run_dir (str): Path to the run directory.
            results (dict): All metrics, including per-epoch history.
        """
        print(f"\n{'=' * 60}")
        print(f"  Run: {self.run_dir}")
        print(f"{'=' * 60}")
        wall_start = time.time()

        # Persist config and parameter counts before training starts.
        save_json(os.path.join(self.run_dir, "config.json"), self.config.to_dict())
        save_json(
            os.path.join(self.run_dir, "run_manifest.json"),
            collect_run_metadata(self.experiment_id, self.seed, self.variant),
        )
        save_json(os.path.join(self.run_dir, "params.json"), self._parameter_counts())

        if self.log_gradients or self.log_activations:
            self._register_hooks()

        best_val_acc = float("-inf")
        history = []
        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            self._grad_norms = []
            self._activation_stats = []

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.config.max_epochs}",
                leave=False,
            )

            for images, labels in pbar:
                images = images.to(self.config.device)
                labels = labels.to(self.config.device)

                logits, loss = self.model(images, labels)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                epoch_loss += loss.item() * labels.size(0)
                epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
                epoch_total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()

            train_acc = epoch_correct / max(epoch_total, 1)
            avg_loss = epoch_loss / max(epoch_total, 1)

            val_metrics = self.evaluate(self.val_loader)
            gen_gap = generalization_gap(train_acc, val_metrics["accuracy"])

            # Gradient diagnostics
            grad_norm_mean = (
                float(np.mean(self._grad_norms)) if self._grad_norms else 0.0
            )
            grad_norm_std = float(np.std(self._grad_norms)) if self._grad_norms else 0.0

            # Activation diagnostics
            if self._activation_stats:
                act_sat = float(
                    np.mean([s["saturation_frac"] for s in self._activation_stats])
                )
                act_std = float(np.mean([s["std"] for s in self._activation_stats]))
            else:
                act_sat = act_std = 0.0

            # TensorBoard
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
            self.writer.add_scalar("Gap/generalization", gen_gap, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            if self.log_gradients:
                self.writer.add_scalar(
                    "Diagnostics/grad_norm_mean", grad_norm_mean, epoch
                )
                self.writer.add_scalar(
                    "Diagnostics/grad_norm_std", grad_norm_std, epoch
                )
            if self.log_activations:
                self.writer.add_scalar(
                    "Diagnostics/activation_saturation", act_sat, epoch
                )
                self.writer.add_scalar("Diagnostics/activation_std", act_std, epoch)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": round(avg_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_metrics["loss"], 4),
                    "val_acc": round(val_metrics["accuracy"], 4),
                    "val_auc": round(val_metrics["auc"], 4),
                    "val_f1": round(val_metrics["f1"], 4),
                    "gen_gap": round(gen_gap, 4),
                    "grad_norm_mean": round(grad_norm_mean, 6),
                    "grad_norm_std": round(grad_norm_std, 6),
                    "act_saturation": round(act_sat, 4),
                    "epoch_time_seconds": round(time.time() - epoch_start, 3),
                }
            )

            print(
                f"  Epoch {epoch:3d} | "
                f"Train: {train_acc:.4f} | Val: {val_metrics['accuracy']:.4f} | "
                f"Gap: {gen_gap:+.4f} | "
                f"Grad: {grad_norm_mean:.4f}±{grad_norm_std:.4f} | "
                f"Sat: {act_sat:.3f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.run_dir, "best_model.pth"),
                )

        total_time = time.time() - start_time

        # Preserve the final optimisation state separately from the selected
        # best-validation checkpoint used for reported test metrics.
        torch.save(
            self.model.state_dict(), os.path.join(self.run_dir, "final_model.pth")
        )

        # Load best checkpoint for final test evaluation.
        evaluation_start = time.time()
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.run_dir, "best_model.pth"),
                map_location=self.config.device,
                weights_only=True,
            )
        )
        test_metrics = self.evaluate(self.test_loader)
        train_final = self.evaluate(self.train_loader)
        final_gen_gap = generalization_gap(
            train_final["accuracy"], test_metrics["accuracy"]
        )

        print(
            f"\n  Test Acc: {test_metrics['accuracy']:.4f} | "
            f"Test AUC: {test_metrics['auc']:.4f} | "
            f"Final Gen Gap: {final_gen_gap:+.4f}"
        )

        noise_results = self._test_noise_robustness() if self.evaluate_noise else {}
        evaluation_time = time.time() - evaluation_start
        total_wall_time = time.time() - wall_start

        results = {
            "schema_version": "1.0",
            "experiment": self.experiment_id,
            "variant": self.variant,
            "dataset": self.config.dataset_name,
            "seed": self.seed,
            "total_time": round(total_time, 1),
            "training_time_seconds": round(total_time, 3),
            "evaluation_time_seconds": round(evaluation_time, 3),
            "total_wall_time_seconds": round(total_wall_time, 3),
            "mean_epoch_time": round(total_time / max(self.config.max_epochs, 1), 3),
            "best_val_acc": round(best_val_acc, 4),
            "train_loss_final": round(train_final["loss"], 4),
            "test_loss": round(test_metrics["loss"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "test_auc": round(test_metrics["auc"], 4),
            "test_f1": round(test_metrics["f1"], 4),
            "train_accuracy_final": round(train_final["accuracy"], 4),
            "generalization_gap": round(final_gen_gap, 4),
            "noise_robustness": noise_results,
            "history": history,
        }
        results["params"] = self._parameter_counts()

        save_json(os.path.join(self.run_dir, "results.json"), results)
        self.writer.close()

        print(f"  Training completed in {total_time:.1f}s → {self.run_dir}")
        return self.run_dir, results
