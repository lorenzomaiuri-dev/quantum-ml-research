import argparse
import json
import math
import os
from datetime import datetime

import numpy as np

from quantum_framework.evaluation import paired_comparison, summarize
from quantum_framework.utils import save_json
from src.config import AblationConfig
from src.engine.trainer import Trainer
from src.models.vit import match_shared_initialization


MODEL_TYPES = ["vanilla", "bounded_mlp", "quantum_reg"]
MODEL_LABELS = {
    "vanilla": "A. Vanilla ViT",
    "bounded_mlp": "B. Bounded MLP",
    "quantum_reg": "C. Quantum-Reg",
}


def _make_config(args, *, dataset=None):
    """Create AblationConfig from parsed CLI args."""
    config = AblationConfig(
        dataset_name=dataset or args.dataset,
        max_epochs=args.epochs,
        embed_dim=args.embed_dim,
        n_head=args.n_head,
        n_layer=args.n_layer,
        ffn_dim=args.ffn_dim,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        q_device=args.q_device,
        eval_interval=args.eval_interval,
    )
    if args.force_cpu:
        config.device = "cpu"
    return config


def _find_run(runs, seed):
    """Return the completed run for a seed, if present."""
    return next((run for run in runs if run["seed"] == seed), None)


def run_train(args):
    """Train a single model."""
    config = _make_config(args)
    trainer = Trainer(
        config,
        model_type=args.model,
        seed=args.seed,
        evaluate_noise=not args.skip_noise,
    )
    _, results = trainer.train()
    print_single_result(results)


def run_compare(args):
    """Train all 3 models on one dataset, one seed. Side-by-side comparison."""
    trainers = {
        model_type: Trainer(
            _make_config(args),
            model_type=model_type,
            seed=args.seed,
            evaluate_noise=not args.skip_noise,
        )
        for model_type in MODEL_TYPES
    }
    shared = match_shared_initialization(
        {name: trainer.model for name, trainer in trainers.items()}
    )
    pairing = {"dataset": args.dataset, "seed": args.seed, **shared}
    results = {}
    for model_type, trainer in trainers.items():
        trainer.pairing = pairing
        _, result = trainer.train()
        results[model_type] = result

    print_comparison(results, args.dataset)

    # Save comparison
    os.makedirs("experiments", exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "experiment": "03_quantum_reg",
        "dataset": args.dataset,
        "seeds": [args.seed],
        "runs": results,
        "paired_statistics": build_paired_statistics(
            {args.dataset: {model: [run] for model, run in results.items()}},
            bootstrap_seed=args.seed,
        )[args.dataset],
    }
    save_json(f"experiments/comparison_{args.dataset}.json", payload)


def run_ablation(args):
    """
    Full ablation study: 3 models × N seeds × M datasets.
    Computes mean ± std across seeds for each model-dataset pair.
    """
    seeds = args.seeds or [42, 137, 256, 512, 1024]
    datasets = args.datasets or ["pathmnist", "bloodmnist", "dermamnist"]

    campaign_name = args.name or "thesis_regularization"
    progress_path = os.path.join(
        "experiments", f"ablation_{campaign_name}.progress.json"
    )
    final_path = os.path.join("experiments", f"ablation_{campaign_name}.json")
    protocol_config = {
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "ffn_dim": args.ffn_dim,
        "batch_size": args.batch_size,
        "train_subset": args.train_subset,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "q_device": args.q_device,
        "device": _make_config(args).device,
        "evaluate_noise": not args.skip_noise,
        "eval_interval": args.eval_interval,
    }
    campaign = {
        "schema_version": "1.1",
        "experiment": "03_quantum_reg",
        "status": "in_progress",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": campaign_name,
        "seeds": seeds,
        "datasets": datasets,
        "config": protocol_config,
        "runs": {
            dataset: {model_type: [] for model_type in MODEL_TYPES}
            for dataset in datasets
        },
        "active_runs": {},
        "protocol": {
            "variant_order": MODEL_TYPES,
            "shared_initialization": True,
            "matched_compression_layer": True,
            "seeded_train_loader_per_block": True,
            "seeded_training_rng_per_variant": True,
            "epoch_level_resume": True,
            "best_validation_checkpoint_for_test": True,
            "legacy_artifacts_excluded": True,
        },
    }

    if args.resume:
        resume_path = progress_path if args.resume == "auto" else args.resume
        with open(resume_path, encoding="utf-8") as stream:
            campaign = json.load(stream)
        if campaign.get("schema_version") != "1.1" or "config" not in campaign:
            raise ValueError(
                "Legacy ablation files cannot be resumed into the definitive campaign"
            )
        if campaign["config"] != protocol_config:
            raise ValueError("Resume requires exactly the original configuration")
        if campaign["seeds"] != seeds or campaign["datasets"] != datasets:
            raise ValueError("Resume requires the original seed and dataset lists")
        campaign["status"] = "in_progress"

    all_results = campaign["runs"]

    def save_progress():
        save_json(progress_path, campaign)

    total_runs = len(MODEL_TYPES) * len(seeds) * len(datasets)
    run_idx = 0

    for dataset in datasets:
        for seed in seeds:
            if all(_find_run(all_results[dataset][model], seed) for model in MODEL_TYPES):
                print(f"  SKIP complete block | {dataset} | seed={seed}")
                run_idx += len(MODEL_TYPES)
                continue

            block_key = f"{dataset}:{seed}"
            active = campaign["active_runs"].setdefault(block_key, {})
            trainers = {}
            for model_type in MODEL_TYPES:
                config = _make_config(args, dataset=dataset)
                trainers[model_type] = Trainer(
                    config,
                    model_type=model_type,
                    seed=seed,
                    run_tag=f"{campaign_name}_{model_type}_{dataset}_s{seed}",
                    run_dir=active.get(model_type),
                    evaluate_noise=not args.skip_noise,
                )
                active[model_type] = trainers[model_type].run_dir

            shared = match_shared_initialization(
                {name: trainer.model for name, trainer in trainers.items()}
            )
            pairing = {"dataset": dataset, "seed": seed, **shared}
            for trainer in trainers.values():
                trainer.pairing = pairing
            save_progress()

            for model_type in MODEL_TYPES:
                run_idx += 1
                if _find_run(all_results[dataset][model_type], seed):
                    print(f"  SKIP {model_type} | {dataset} | seed={seed} (completed)")
                    continue
                print(f"\n{'#' * 60}")
                print(
                    f"  RUN {run_idx}/{total_runs}: {model_type} | {dataset} | seed={seed}"
                )
                print(f"{'#' * 60}")
                _, result = trainers[model_type].train()
                all_results[dataset][model_type].append(result)
                all_results[dataset][model_type].sort(
                    key=lambda run: seeds.index(run["seed"])
                )
                save_progress()

    # Aggregate and print
    if any(
        _find_run(all_results[dataset][model], seed) is None
        for dataset in datasets
        for model in MODEL_TYPES
        for seed in seeds
    ):
        raise RuntimeError("Campaign ended with incomplete dataset/seed/model blocks")

    print_ablation_summary(all_results, seeds)
    statistics = build_paired_statistics(all_results, bootstrap_seed=seeds[0])

    # Save full results
    os.makedirs("experiments", exist_ok=True)
    # Convert defaultdict to regular dict for JSON
    serializable = {
        ds: {mt: runs for mt, runs in models.items()}
        for ds, models in all_results.items()
    }
    campaign["runs"] = serializable
    campaign["paired_statistics"] = statistics
    campaign["status"] = "complete"
    campaign["completed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    save_json(final_path, campaign)
    save_json(progress_path, campaign)


def run_screen(args):
    """Select an informative scarce-data regime without evaluating the test set."""
    seeds = args.seeds or [42, 137, 256]
    datasets = args.datasets or ["pathmnist", "bloodmnist", "dermamnist"]
    subsets = args.subsets or [100, 250, 500, 1000]
    campaign_name = args.name or "classical_regime_screen"
    progress_path = os.path.join("experiments", f"{campaign_name}.progress.json")
    final_path = os.path.join("experiments", f"{campaign_name}.json")
    protocol_config = {
        "target_optimizer_steps": args.target_steps,
        "validation_checkpoints": args.validation_checkpoints,
        "embed_dim": args.embed_dim,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
        "ffn_dim": args.ffn_dim,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "device": _make_config(args).device,
        "model_type": "bounded_mlp",
        "test_set_evaluated": False,
    }
    campaign = {
        "schema_version": "1.0",
        "experiment": "03_quantum_reg_regime_screen",
        "status": "in_progress",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": campaign_name,
        "seeds": seeds,
        "datasets": datasets,
        "subsets": subsets,
        "config": protocol_config,
        "runs": [],
        "active_runs": {},
        "selection_rule": (
            "Choose one common subset using only bounded-MLP train/validation "
            "metrics; require positive validation gap and adequate train fit."
        ),
    }
    if args.resume:
        with open(progress_path, encoding="utf-8") as stream:
            campaign = json.load(stream)
        if campaign["config"] != protocol_config:
            raise ValueError("Screen resume requires exactly the original configuration")
        if (
            campaign["seeds"] != seeds
            or campaign["datasets"] != datasets
            or campaign["subsets"] != subsets
        ):
            raise ValueError("Screen resume requires the original sweep grid")
        campaign["status"] = "in_progress"

    def find_screen_run(dataset, subset, seed):
        return next(
            (
                run
                for run in campaign["runs"]
                if run["dataset"] == dataset
                and run["train_subset"] == subset
                and run["seed"] == seed
            ),
            None,
        )

    for dataset in datasets:
        for subset in subsets:
            for seed in seeds:
                if find_screen_run(dataset, subset, seed):
                    continue
                key = f"{dataset}:{subset}:{seed}"
                config = _make_config(args, dataset=dataset)
                config.train_subset = subset
                steps_per_epoch = math.ceil(subset / config.batch_size)
                config.max_epochs = math.ceil(args.target_steps / steps_per_epoch)
                config.eval_interval = max(
                    1,
                    round(config.max_epochs / args.validation_checkpoints),
                )
                trainer = Trainer(
                    config,
                    model_type="bounded_mlp",
                    seed=seed,
                    run_tag=f"{campaign_name}_{dataset}_n{subset}_s{seed}",
                    run_dir=campaign["active_runs"].get(key),
                    evaluate_noise=False,
                    evaluate_test=False,
                )
                campaign["active_runs"][key] = trainer.run_dir
                save_json(progress_path, campaign)
                _, result = trainer.train()
                result["train_subset"] = subset
                result["screen_schedule"] = {
                    "steps_per_epoch": steps_per_epoch,
                    "epochs": config.max_epochs,
                    "planned_optimizer_steps": steps_per_epoch * config.max_epochs,
                    "eval_interval": config.eval_interval,
                }
                campaign["runs"].append(result)
                campaign["runs"].sort(
                    key=lambda run: (
                        datasets.index(run["dataset"]),
                        subsets.index(run["train_subset"]),
                        seeds.index(run["seed"]),
                    )
                )
                save_json(progress_path, campaign)

    campaign["status"] = "complete"
    campaign["completed_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    save_json(final_path, campaign)
    save_json(progress_path, campaign)


def build_paired_statistics(all_results, bootstrap_seed=42):
    """Build thesis-ready summaries with seed-aligned paired comparisons."""
    output = {}
    metrics = (
        "test_accuracy",
        "test_auc",
        "test_f1",
        "generalization_gap",
        "total_time",
    )
    for dataset, models in all_results.items():
        output[dataset] = {"summaries": {}, "comparisons": {}}
        for model_type, runs in models.items():
            output[dataset]["summaries"][model_type] = {
                metric: summarize([run[metric] for run in runs]) for metric in metrics
            }

        for baseline in ("bounded_mlp", "vanilla"):
            comparison_name = f"quantum_reg_vs_{baseline}"
            output[dataset]["comparisons"][comparison_name] = {}
            q_by_seed = {run["seed"]: run for run in models["quantum_reg"]}
            c_by_seed = {run["seed"]: run for run in models[baseline]}
            common_seeds = sorted(q_by_seed.keys() & c_by_seed.keys())
            for metric in metrics:
                result = paired_comparison(
                    [q_by_seed[seed][metric] for seed in common_seeds],
                    [c_by_seed[seed][metric] for seed in common_seeds],
                    seed=bootstrap_seed,
                )
                result["seeds"] = common_seeds
                output[dataset]["comparisons"][comparison_name][metric] = result
    return output


def print_single_result(r):
    print(f"\n{'=' * 50}")
    print(f"  {MODEL_LABELS.get(r['model_type'], r['model_type'])}")
    print(f"  Test Acc:  {r['test_accuracy']:.4f}")
    print(f"  Test AUC:  {r['test_auc']:.4f}")
    print(f"  Gen Gap:   {r['generalization_gap']:+.4f}")
    print(f"  Params:    {r['params']['total']:,}")
    print(f"  Time:      {r['total_time']:.1f}s")
    print(f"{'=' * 50}")


def print_comparison(results, dataset):
    print(f"\n{'=' * 70}")
    print(f"  COMPARISON: {dataset}")
    print(f"{'=' * 70}")
    print(
        f"{'Model':<22} {'Test Acc':>10} {'AUC':>8} {'Gen Gap':>10} {'Params':>10} {'Time':>8}"
    )
    print(f"{'-' * 68}")
    for mt in MODEL_TYPES:
        r = results[mt]
        label = MODEL_LABELS[mt]
        print(
            f"{label:<22} "
            f"{r['test_accuracy']:>10.4f} "
            f"{r['test_auc']:>8.4f} "
            f"{r['generalization_gap']:>+10.4f} "
            f"{r['params']['total']:>10,} "
            f"{r['total_time']:>7.1f}s"
        )
    print(f"{'=' * 70}")


def print_ablation_summary(all_results, seeds):
    """Print aggregated results across seeds with mean ± std."""
    print(f"\n{'=' * 80}")
    print(f"  ABLATION STUDY SUMMARY ({len(seeds)} seeds per configuration)")
    print(f"{'=' * 80}")

    for dataset, models in all_results.items():
        print(f"\n  --- {dataset} ---")
        print(
            f"  {'Model':<22} {'Test Acc':>14} {'Gen Gap':>14} {'AUC':>14} {'Noise 0.1':>12}"
        )
        print(f"  {'-' * 76}")

        for mt in MODEL_TYPES:
            runs = models[mt]
            accs = [r["test_accuracy"] for r in runs]
            gaps = [r["generalization_gap"] for r in runs]
            aucs = [r["test_auc"] for r in runs]

            # Noise robustness at intensity 0.1 (gaussian)
            noise_accs = []
            for r in runs:
                nr = r.get("noise_robustness", {}).get("gaussian", {})
                if "0.1" in nr:
                    value = nr["0.1"]
                    noise_accs.append(
                        value["accuracy"] if isinstance(value, dict) else value
                    )

            label = MODEL_LABELS[mt]
            acc_str = f"{np.mean(accs):.4f}±{np.std(accs):.4f}"
            gap_str = f"{np.mean(gaps):+.4f}±{np.std(gaps):.4f}"
            auc_str = f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}"
            noise_str = (
                f"{np.mean(noise_accs):.4f}±{np.std(noise_accs):.4f}"
                if noise_accs
                else "N/A"
            )

            print(
                f"  {label:<22} {acc_str:>14} {gap_str:>14} {auc_str:>14} {noise_str:>12}"
            )

    print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantum Regularization Ablation Study"
    )
    sub = parser.add_subparsers(dest="mode", help="Operating mode")

    # Shared args
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--dataset", default="pathmnist")
    shared.add_argument("--epochs", type=int, default=50)
    shared.add_argument("--embed-dim", type=int, default=10)
    shared.add_argument("--n-head", type=int, default=2)
    shared.add_argument("--n-layer", type=int, default=2)
    shared.add_argument("--ffn-dim", type=int, default=40)
    shared.add_argument("--batch-size", type=int, default=128)
    shared.add_argument("--dropout", type=float, default=0.1)
    shared.add_argument("--weight-decay", type=float, default=1e-4)
    shared.add_argument("--q-device", default="default.qubit")
    shared.add_argument("--force-cpu", "--force_cpu", action="store_true")
    shared.add_argument("--skip-noise", action="store_true")
    shared.add_argument("--eval-interval", type=int, default=5)
    shared.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="Stratified subset size (0=full). Use 2000-5000 for overfitting regime",
    )

    # Train
    train_p = sub.add_parser("train", parents=[shared])
    train_p.add_argument(
        "--model",
        choices=MODEL_TYPES,
        default="quantum_reg",
    )
    train_p.add_argument("--seed", type=int, default=42)

    # Compare
    compare_p = sub.add_parser("compare", parents=[shared])
    compare_p.add_argument("--seed", type=int, default=42)

    # Ablation
    abl_p = sub.add_parser("ablation", parents=[shared])
    abl_p.add_argument("--seeds", type=int, nargs="+", default=None)
    abl_p.add_argument("--datasets", type=str, nargs="+", default=None)
    abl_p.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume the definitive campaign (optionally from an explicit path)",
    )
    abl_p.add_argument("--name", default=None)

    # Classical-only protocol selection; deliberately never touches the test set.
    screen_p = sub.add_parser("screen", parents=[shared])
    screen_p.add_argument("--seeds", type=int, nargs="+", default=None)
    screen_p.add_argument("--datasets", type=str, nargs="+", default=None)
    screen_p.add_argument("--subsets", type=int, nargs="+", default=None)
    screen_p.add_argument("--name", default=None)
    screen_p.add_argument("--resume", action="store_true")
    screen_p.add_argument("--target-steps", type=int, default=400)
    screen_p.add_argument("--validation-checkpoints", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "ablation":
        run_ablation(args)
    elif args.mode == "screen":
        run_screen(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
