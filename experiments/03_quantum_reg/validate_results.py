"""Audit and re-evaluate checkpoints from a complete experiment 03 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from quantum_framework.evaluation import compute_classification_metrics
from quantum_framework.utils import save_json
from src.config import AblationConfig
from src.data.loaders import load_dataset
from src.models.vit import AblationViT


VARIANTS = ("vanilla", "bounded_mlp", "quantum_reg")
REQUIRED_ARTIFACTS = (
    "best_model.pth",
    "config.json",
    "final_model.pth",
    "latest_checkpoint.pth",
    "params.json",
    "results.json",
    "run_manifest.json",
)
METRIC_TOLERANCE = 5.1e-5


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_run_dir(aggregate_path: Path, recorded_path: str) -> Path:
    relative = Path(recorded_path)
    candidates = (
        aggregate_path.parent.parent / relative,
        Path(__file__).resolve().parent / relative,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"cannot resolve run directory {recorded_path!r} from {aggregate_path}"
    )


def _load_config(run_dir: Path) -> AblationConfig:
    raw = _read_json(run_dir / "config.json")
    init_fields = {field.name for field in fields(AblationConfig) if field.init}
    config = AblationConfig(**{key: raw[key] for key in init_fields if key in raw})
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    return config


@torch.inference_mode()
def _evaluate(model: AblationViT, loader, device: str) -> dict[str, float]:
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


def _assert_close(
    *, dataset: str, variant: str, seed: int, errors: dict[str, float]
) -> None:
    failures = {
        metric: error
        for metric, error in errors.items()
        if not math.isfinite(error) or abs(error) > METRIC_TOLERANCE
    }
    if failures:
        raise RuntimeError(
            f"checkpoint mismatch for {dataset}/{variant}/seed={seed}: {failures}"
        )


def _audit_record(
    aggregate_path: Path,
    dataset: str,
    variant: str,
    recorded: dict[str, Any],
) -> dict[str, Any]:
    seed = recorded["seed"]
    run_dir = _resolve_run_dir(aggregate_path, recorded["run_dir"])
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing artifacts in {run_dir}: {missing}")

    persisted = _read_json(run_dir / "results.json")
    if persisted != recorded:
        raise RuntimeError(f"aggregate/results.json mismatch in {run_dir}")
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("pairing") != recorded.get("pairing"):
        raise RuntimeError(f"manifest pairing mismatch in {run_dir}")

    config_raw = _read_json(run_dir / "config.json")
    max_epochs = int(config_raw["max_epochs"])
    eval_interval = max(int(config_raw.get("eval_interval", 1)), 1)
    history = recorded.get("history", [])
    expected_epochs = list(range(1, max_epochs + 1))
    if [point.get("epoch") for point in history] != expected_epochs:
        raise RuntimeError(f"invalid epoch history in {run_dir}")
    validation_epochs = [
        point["epoch"] for point in history if point.get("val_acc") is not None
    ]
    expected_validation_epochs = sorted(
        {1, max_epochs, *range(eval_interval, max_epochs + 1, eval_interval)}
    )
    if validation_epochs != expected_validation_epochs:
        raise RuntimeError(
            f"invalid validation cadence in {run_dir}: {validation_epochs}"
        )

    pairing = recorded.get("pairing", {})
    if pairing.get("dataset") != dataset or pairing.get("seed") != seed:
        raise RuntimeError(f"invalid pairing identity in {run_dir}")
    if len(pairing.get("sha256", "")) != 64:
        raise RuntimeError(f"invalid pairing digest in {run_dir}")

    return {
        "seed": seed,
        "run_dir": recorded["run_dir"],
        "pairing_sha256": pairing["sha256"],
        "artifact_sha256": {
            name: _sha256(run_dir / name) for name in REQUIRED_ARTIFACTS
        },
        "manifest_verified": True,
        "history_verified": True,
    }


def _recompute_record(
    aggregate_path: Path,
    variant: str,
    recorded: dict[str, Any],
) -> dict[str, Any]:
    run_dir = _resolve_run_dir(aggregate_path, recorded["run_dir"])
    config = _load_config(run_dir)
    train_loader, val_loader, test_loader = load_dataset(config, seed=recorded["seed"])
    model = AblationViT(config, variant).to(config.device)
    model.load_state_dict(
        torch.load(
            run_dir / "best_model.pth",
            map_location=config.device,
            weights_only=True,
        )
    )
    recomputed = {
        "train": _evaluate(model, train_loader, config.device),
        "validation": _evaluate(model, val_loader, config.device),
        "test": _evaluate(model, test_loader, config.device),
    }
    expected = {
        "train_loss": recorded["train_loss_final"],
        "train_accuracy": recorded["train_accuracy_final"],
        "validation_loss": recorded["validation_loss"],
        "validation_accuracy": recorded["validation_accuracy"],
        "validation_auc": recorded["validation_auc"],
        "validation_f1": recorded["validation_f1"],
        "test_loss": recorded["test_loss"],
        "test_accuracy": recorded["test_accuracy"],
        "test_auc": recorded["test_auc"],
        "test_f1": recorded["test_f1"],
        "generalization_gap": recorded["generalization_gap"],
    }
    actual = {
        "train_loss": recomputed["train"]["loss"],
        "train_accuracy": recomputed["train"]["accuracy"],
        "validation_loss": recomputed["validation"]["loss"],
        "validation_accuracy": recomputed["validation"]["accuracy"],
        "validation_auc": recomputed["validation"]["auc"],
        "validation_f1": recomputed["validation"]["f1"],
        "test_loss": recomputed["test"]["loss"],
        "test_accuracy": recomputed["test"]["accuracy"],
        "test_auc": recomputed["test"]["auc"],
        "test_f1": recomputed["test"]["f1"],
        "generalization_gap": (
            recomputed["train"]["accuracy"] - recomputed["test"]["accuracy"]
        ),
    }
    errors = {metric: actual[metric] - value for metric, value in expected.items()}
    _assert_close(
        dataset=config.dataset_name,
        variant=variant,
        seed=recorded["seed"],
        errors=errors,
    )
    return {"expected": expected, "recomputed": recomputed, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Verify files, manifests, histories, and pairing without inference.",
    )
    args = parser.parse_args()

    aggregate = _read_json(args.aggregate)
    if aggregate.get("experiment") != "03_quantum_reg":
        raise ValueError("expected an experiment 03 aggregate")
    if aggregate.get("status") != "complete":
        raise ValueError("aggregate must be complete")
    if aggregate.get("schema_version") != "1.1":
        raise ValueError("expected campaign schema 1.1")

    seeds = aggregate["seeds"]
    output: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": "03_quantum_reg_checkpoint_validation",
        "source": str(args.aggregate),
        "source_sha256": _sha256(args.aggregate),
        "mode": "audit_only" if args.audit_only else "full_recomputation",
        "metric_tolerance": METRIC_TOLERANCE,
        "datasets": aggregate["datasets"],
        "seeds": seeds,
        "runs": {},
    }
    for dataset in aggregate["datasets"]:
        output["runs"][dataset] = {}
        pairing_by_seed: dict[int, set[str]] = {seed: set() for seed in seeds}
        for variant in VARIANTS:
            records = aggregate["runs"][dataset][variant]
            if [record["seed"] for record in records] != seeds:
                raise RuntimeError(f"incomplete or unordered {dataset}/{variant} runs")
            output["runs"][dataset][variant] = []
            for recorded in records:
                audit = _audit_record(args.aggregate, dataset, variant, recorded)
                pairing_by_seed[recorded["seed"]].add(audit["pairing_sha256"])
                if not args.audit_only:
                    audit["checkpoint_validation"] = _recompute_record(
                        args.aggregate, variant, recorded
                    )
                output["runs"][dataset][variant].append(audit)
                print(f"validated {dataset}/{variant}/seed={recorded['seed']}")
        mismatched = {
            seed: sorted(digests)
            for seed, digests in pairing_by_seed.items()
            if len(digests) != 1
        }
        if mismatched:
            raise RuntimeError(f"pairing SHA mismatch for {dataset}: {mismatched}")

    output["status"] = "valid"
    save_json(str(args.output), output)
    print(f"validation complete: {args.output}")


if __name__ == "__main__":
    main()
