import argparse
import json
import os
from collections import defaultdict

import numpy as np

from src.config import AblationConfig
from src.engine.trainer import Trainer


MODEL_TYPES = ["vanilla", "bounded_mlp", "quantum_reg"]
MODEL_LABELS = {
    "vanilla": "A. Vanilla ViT",
    "bounded_mlp": "B. Bounded MLP",
    "quantum_reg": "C. Quantum-Reg",
}


def _make_config(args):
    """Create AblationConfig from parsed CLI args."""
    return AblationConfig(
        dataset_name=args.dataset,
        max_epochs=args.epochs,
        embed_dim=args.embed_dim,
        n_head=args.n_head,
        n_layer=args.n_layer,
        ffn_dim=args.ffn_dim,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )


def run_train(args):
    """Train a single model."""
    config = _make_config(args)
    trainer = Trainer(config, model_type=args.model, seed=args.seed)
    _, results = trainer.train()
    print_single_result(results)


def run_compare(args):
    """Train all 3 models on one dataset, one seed. Side-by-side comparison."""
    results = {}

    for model_type in MODEL_TYPES:
        config = _make_config(args)
        trainer = Trainer(config, model_type=model_type, seed=args.seed)
        _, result = trainer.train()
        results[model_type] = result

    print_comparison(results, args.dataset)

    # Save comparison
    os.makedirs("experiments", exist_ok=True)
    with open(f"experiments/comparison_{args.dataset}.json", "w") as f:
        json.dump(results, f, indent=4)


def run_ablation(args):
    """
    Full ablation study: 3 models × N seeds × M datasets.
    Computes mean ± std across seeds for each model-dataset pair.
    """
    seeds = args.seeds or [42, 137, 256, 512, 1024]
    datasets = args.datasets or ["pathmnist", "bloodmnist", "dermamnist"]

    all_results = defaultdict(lambda: defaultdict(list))

    total_runs = len(MODEL_TYPES) * len(seeds) * len(datasets)
    run_idx = 0

    for dataset in datasets:
        for model_type in MODEL_TYPES:
            for seed in seeds:
                run_idx += 1
                print(f"\n{'#'*60}")
                print(f"  RUN {run_idx}/{total_runs}: {model_type} | {dataset} | seed={seed}")
                print(f"{'#'*60}")

                config = _make_config(args)
                config.dataset_name = dataset
                trainer = Trainer(
                    config,
                    model_type=model_type,
                    seed=seed,
                    run_tag=f"abl_{model_type}_{dataset}_s{seed}",
                )
                _, result = trainer.train()
                all_results[dataset][model_type].append(result)

    # Aggregate and print
    print_ablation_summary(all_results, seeds)

    # Save full results
    os.makedirs("experiments", exist_ok=True)
    # Convert defaultdict to regular dict for JSON
    serializable = {
        ds: {mt: runs for mt, runs in models.items()}
        for ds, models in all_results.items()
    }
    with open("experiments/ablation_full_results.json", "w") as f:
        json.dump(serializable, f, indent=4)


def print_single_result(r):
    print(f"\n{'='*50}")
    print(f"  {MODEL_LABELS.get(r['model_type'], r['model_type'])}")
    print(f"  Test Acc:  {r['test_accuracy']:.4f}")
    print(f"  Test AUC:  {r['test_auc']:.4f}")
    print(f"  Gen Gap:   {r['generalization_gap']:+.4f}")
    print(f"  Params:    {r['params']['total']:,}")
    print(f"  Time:      {r['total_time']:.1f}s")
    print(f"{'='*50}")


def print_comparison(results, dataset):
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {dataset}")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Test Acc':>10} {'AUC':>8} {'Gen Gap':>10} {'Params':>10} {'Time':>8}")
    print(f"{'-'*68}")
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
    print(f"{'='*70}")


def print_ablation_summary(all_results, seeds):
    """Print aggregated results across seeds with mean ± std."""
    print(f"\n{'='*80}")
    print(f"  ABLATION STUDY SUMMARY ({len(seeds)} seeds per configuration)")
    print(f"{'='*80}")

    for dataset, models in all_results.items():
        print(f"\n  --- {dataset} ---")
        print(f"  {'Model':<22} {'Test Acc':>14} {'Gen Gap':>14} {'AUC':>14} {'Noise 0.1':>12}")
        print(f"  {'-'*76}")

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
                    noise_accs.append(nr["0.1"])

            label = MODEL_LABELS[mt]
            acc_str = f"{np.mean(accs):.4f}±{np.std(accs):.4f}"
            gap_str = f"{np.mean(gaps):+.4f}±{np.std(gaps):.4f}"
            auc_str = f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}"
            noise_str = (
                f"{np.mean(noise_accs):.4f}±{np.std(noise_accs):.4f}"
                if noise_accs
                else "N/A"
            )

            print(f"  {label:<22} {acc_str:>14} {gap_str:>14} {auc_str:>14} {noise_str:>12}")

    print(f"\n{'='*80}")


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
    shared.add_argument(
        "--train-subset", type=int, default=0,
        help="Stratified subset size (0=full). Use 2000-5000 for overfitting regime"
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

    args = parser.parse_args()

    if args.mode == "train":
        run_train(args)
    elif args.mode == "compare":
        run_compare(args)
    elif args.mode == "ablation":
        run_ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()