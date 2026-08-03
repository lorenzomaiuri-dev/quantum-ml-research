"""Re-evaluate best and final checkpoints from experiment 02."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from quantum_framework.evaluation import (
    compute_classification_metrics,
    paired_comparison,
)
from quantum_framework.utils import save_json
from src.config import QViTConfig
from src.data.medmnist import get_medmnist_loaders
from src.models.hybrid_vit import HybridQCNNViT


@torch.no_grad()
def evaluate(model: HybridQCNNViT, loader, device: str) -> dict[str, float]:
    model.eval()
    labels_all, predictions_all, probabilities_all = [], [], []
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, loss = model(images, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        labels_all.extend(labels.cpu().numpy())
        predictions_all.extend(logits.argmax(dim=-1).cpu().numpy())
        probabilities_all.extend(torch.softmax(logits, dim=-1).cpu().numpy())

    metrics = compute_classification_metrics(
        labels_all, predictions_all, probabilities_all
    )
    return {
        "loss": total_loss / total_samples,
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "f1": metrics["f1"],
    }


def load_config(run_dir: Path) -> QViTConfig:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    init_fields = {field.name for field in fields(QViTConfig) if field.init}
    config = QViTConfig(**{key: raw[key] for key in init_fields})
    config.device = "cpu"
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if aggregate.get("experiment") != "02_quantum_vit":
        raise ValueError("expected an experiment 02 aggregate")
    if aggregate.get("status") != "complete":
        raise ValueError("aggregate must be complete")

    output = {
        "schema_version": "1.0",
        "experiment": "02_quantum_vit_checkpoint_validation",
        "source": str(args.aggregate),
        "seeds": aggregate["seeds"],
        "runs": {"classical": [], "quantum": []},
    }
    for variant in ("classical", "quantum"):
        for recorded in aggregate["runs"][variant]:
            run_dir = args.aggregate.parent.parent / recorded["run_dir"]
            config = load_config(run_dir)
            _, _, test_loader = get_medmnist_loaders(config)
            model = HybridQCNNViT(config).to(config.device)
            evaluations = {}
            for checkpoint_name in ("best_model.pth", "final_model.pth"):
                model.load_state_dict(
                    torch.load(
                        run_dir / checkpoint_name,
                        map_location=config.device,
                        weights_only=True,
                    )
                )
                evaluations[checkpoint_name.removesuffix("_model.pth")] = evaluate(
                    model, test_loader, config.device
                )

            expected = {
                "loss": recorded["test_loss"],
                "accuracy": recorded["test_accuracy"],
                "auc": recorded["test_auc"],
                "f1": recorded["test_f1"],
            }
            errors = {
                metric: evaluations["best"][metric] - value
                for metric, value in expected.items()
            }
            if any(abs(error) > 5.1e-5 for error in errors.values()):
                raise RuntimeError(
                    f"best checkpoint mismatch for {variant} seed {recorded['seed']}: "
                    f"{errors}"
                )
            output["runs"][variant].append(
                {
                    "seed": recorded["seed"],
                    "run_dir": recorded["run_dir"],
                    "recorded_best": expected,
                    "recomputed_best": evaluations["best"],
                    "recording_errors": errors,
                    "final": evaluations["final"],
                }
            )

    output["paired_statistics"] = {}
    for checkpoint in ("recomputed_best", "final"):
        output["paired_statistics"][checkpoint] = {}
        for metric in ("accuracy", "auc", "f1"):
            output["paired_statistics"][checkpoint][metric] = paired_comparison(
                [run[checkpoint][metric] for run in output["runs"]["quantum"]],
                [run[checkpoint][metric] for run in output["runs"]["classical"]],
            )

    save_json(str(args.output), output)


if __name__ == "__main__":
    main()
