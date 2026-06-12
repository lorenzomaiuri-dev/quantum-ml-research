import argparse

from src.config import QViTConfig
from src.engine.trainer import Trainer


def run_comparison(args):
    results = []
    for use_quantum in [False, True]:
        mode = "quantum" if use_quantum else "classical"
        config = QViTConfig(
            dataset_name=args.dataset,
            max_epochs=args.epochs,
            use_quantum=use_quantum,
            embed_dim=args.embed_dim,
            n_head=args.n_head,
        )
        _, metrics = Trainer(config).train()
        metrics["mode"] = mode
        results.append(metrics)

    print(f"\n{'='*50}")
    print(f"COMPARISON: {args.dataset.upper()}")
    print(f"{'='*50}")
    for r in results:
        print(f"  {r['mode']:<10} acc={r['test_accuracy']:.4f}  time={r['total_time']:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Quantum ViT — MedMNIST classification")
    parser.add_argument("mode", choices=["train", "compare"])
    parser.add_argument("--dataset", default="pathmnist")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=2)
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
        Trainer(config).train()


if __name__ == "__main__":
    main()
