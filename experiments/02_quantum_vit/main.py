import argparse
import os

from quantum_framework.evaluation import paired_comparison
from quantum_framework.utils import save_json
from src.config import QViTConfig
from src.engine.trainer import Trainer


def run_comparison(args):
    seeds = args.seeds or [args.seed]
    results = {"classical": [], "quantum": []}
    for seed in seeds:
        for use_quantum in [False, True]:
            mode = "quantum" if use_quantum else "classical"
            config = QViTConfig(
                dataset_name=args.dataset,
                max_epochs=args.epochs,
                use_quantum=use_quantum,
                embed_dim=args.embed_dim,
                n_head=args.n_head,
            )
            _, metrics = Trainer(config, seed=seed).train()
            results[mode].append(metrics)

    print(f"\n{'=' * 50}")
    print(f"COMPARISON: {args.dataset.upper()}")
    print(f"{'=' * 50}")
    for mode, runs in results.items():
        for r in runs:
            print(
                f"  {mode:<10} seed={r['seed']:<5} "
                f"acc={r['test_accuracy']:.4f} time={r['total_time']:.1f}s"
            )

    statistics = {}
    for metric in (
        "test_accuracy",
        "test_auc",
        "test_f1",
        "generalization_gap",
        "total_time",
    ):
        statistics[metric] = paired_comparison(
            [r[metric] for r in results["quantum"]],
            [r[metric] for r in results["classical"]],
            seed=args.seed,
        )
    payload = {
        "schema_version": "1.0",
        "experiment": "02_quantum_vit",
        "dataset": args.dataset,
        "seeds": seeds,
        "runs": results,
        "paired_statistics": statistics,
    }
    os.makedirs("experiments", exist_ok=True)
    save_json(f"experiments/comparison_{args.dataset}.json", payload)


def main():
    parser = argparse.ArgumentParser(
        description="Quantum ViT — MedMNIST classification"
    )
    parser.add_argument("mode", choices=["train", "compare"])
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(args)
    else:
        config = QViTConfig(
            dataset_name=args.dataset,
            max_epochs=args.epochs,
            use_quantum=not args.classical,
            embed_dim=args.embed_dim,
            n_head=args.n_head,
        )
        Trainer(config, seed=args.seed).train()


if __name__ == "__main__":
    main()
