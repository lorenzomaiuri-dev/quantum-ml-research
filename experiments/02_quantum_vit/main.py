import argparse
import json
import os
from datetime import datetime

from quantum_framework.evaluation import paired_comparison
from quantum_framework.utils import save_json
from src.config import QViTConfig
from src.engine.trainer import Trainer
from src.models.hybrid_vit import match_shared_initialization


def build_config(args, *, use_quantum: bool) -> QViTConfig:
    config = QViTConfig(
        dataset_name=args.dataset,
        max_epochs=args.epochs,
        use_quantum=use_quantum,
        embed_dim=args.embed_dim,
        n_head=args.n_head,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        q_device=args.q_device,
    )
    if args.force_cpu:
        config.device = "cpu"
    return config


def _find_run(runs: list[dict], seed: int) -> dict | None:
    return next((run for run in runs if run["seed"] == seed), None)


def run_comparison(args) -> None:
    seeds = args.seeds or [args.seed]
    campaign_name = args.name or f"thesis_{args.dataset}"
    progress_path = os.path.join(
        "experiments", f"comparison_{campaign_name}.progress.json"
    )
    final_path = os.path.join("experiments", f"comparison_{campaign_name}.json")
    protocol_config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "n_head": args.n_head,
        "batch_size": args.batch_size,
        "train_subset": args.train_subset,
        "q_device": args.q_device,
        "device": (
            "cpu"
            if args.force_cpu
            else build_config(args, use_quantum=False).device
        ),
    }
    campaign = {
        "schema_version": "1.1",
        "experiment": "02_quantum_vit",
        "status": "in_progress",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": campaign_name,
        "seeds": seeds,
        "config": protocol_config,
        "runs": {"classical": [], "quantum": []},
        "active_runs": {},
        "protocol": {
            "variant_order": ["classical", "quantum"],
            "shared_initialization": True,
            "matched_classical_adapter": True,
            "seeded_train_loader_per_pair": True,
            "seeded_training_rng_per_variant": True,
            "epoch_level_resume": True,
            "best_validation_checkpoint_for_test": True,
        },
    }

    if args.resume:
        if not os.path.exists(progress_path):
            raise FileNotFoundError(f"No progress file found: {progress_path}")
        with open(progress_path, encoding="utf-8") as stream:
            campaign = json.load(stream)
        if campaign["config"] != protocol_config:
            raise ValueError("Resume requires exactly the original configuration")
        if not set(campaign["seeds"]).issubset(seeds):
            raise ValueError(
                "Resume may extend, but not remove, the original seed list"
            )
        campaign["seeds"] = seeds
        campaign["status"] = "in_progress"

    os.makedirs("experiments", exist_ok=True)

    def save_progress() -> None:
        save_json(progress_path, campaign)

    results = campaign["runs"]
    for seed in seeds:
        pair_complete = all(
            _find_run(results[variant], seed)
            for variant in ("classical", "quantum")
        )
        if pair_complete:
            print(f"Skipping completed pair seed {seed}")
            continue

        seed_key = str(seed)
        active = campaign["active_runs"].setdefault(seed_key, {})
        trainers = {}
        for variant, use_quantum in (("classical", False), ("quantum", True)):
            config = build_config(args, use_quantum=use_quantum)
            trainers[variant] = Trainer(
                config,
                seed=seed,
                experiment_name=f"{campaign_name}_{variant}_s{seed}",
                run_dir=active.get(variant),
            )
            active[variant] = trainers[variant].run_dir

        shared = match_shared_initialization(
            trainers["classical"].model, trainers["quantum"].model
        )
        pairing = {
            "seed": seed,
            "shared_state_sha256": shared["sha256"],
            "matched_state_keys": shared["matched_keys"],
        }
        for trainer in trainers.values():
            trainer.pairing = pairing
        save_progress()

        for variant in ("classical", "quantum"):
            if _find_run(results[variant], seed):
                print(f"Skipping completed {variant} seed {seed}")
                continue
            _, metrics = trainers[variant].train()
            results[variant].append(metrics)
            results[variant].sort(key=lambda run: seeds.index(run["seed"]))
            save_progress()

    ordered = {
        variant: [_find_run(results[variant], seed) for seed in seeds]
        for variant in ("classical", "quantum")
    }
    if any(run is None for runs in ordered.values() for run in runs):
        raise RuntimeError("Campaign ended with incomplete pairs")

    statistics = {}
    for metric in (
        "test_accuracy",
        "test_auc",
        "test_f1",
        "generalization_gap",
        "training_time_seconds",
        "evaluation_time_seconds",
    ):
        statistics[metric] = paired_comparison(
            [run[metric] for run in ordered["quantum"]],
            [run[metric] for run in ordered["classical"]],
            seed=args.seed,
        )
    statistics["parameter_count"] = paired_comparison(
        [run["params"]["trainable"] for run in ordered["quantum"]],
        [run["params"]["trainable"] for run in ordered["classical"]],
        seed=args.seed,
    )
    campaign["paired_statistics"] = statistics
    campaign["status"] = "complete"
    campaign["completed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    save_json(final_path, campaign)
    save_json(progress_path, campaign)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantum ViT — MedMNIST classification"
    )
    parser.add_argument("mode", choices=["train", "compare"])
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--q-device", default="default.qubit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-cpu", "--force_cpu", action="store_true")
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(args)
    else:
        config = build_config(args, use_quantum=not args.classical)
        Trainer(config, seed=args.seed).train()


if __name__ == "__main__":
    main()
