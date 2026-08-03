import argparse
import importlib
import secrets
import os
import json
import shutil
from datetime import datetime
from quantum_framework.evaluation import paired_comparison
from quantum_framework.utils import set_seed
from quantum_framework.utils import save_json
from src.model import match_shared_initialization
from src.training import Trainer
from src.inference import Generator


def main():
    parser = argparse.ArgumentParser(description="Quantum GPT Runner")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "generate", "full", "compare"],
    )
    parser.add_argument("--name", type=str, default="quantum_gpt")
    parser.add_argument("--dataset", type=str, default="input.txt")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--print_in_place", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compare campaign from comparison_<name>.progress.json",
    )
    implementation = parser.add_mutually_exclusive_group()
    implementation.add_argument("--quantum", action="store_true")
    implementation.add_argument("--classical", action="store_true")
    parser.add_argument("--hint", type=str, default="", nargs="+")

    args = parser.parse_args()
    args.hint = " ".join(args.hint)

    try:
        config_module = importlib.import_module(f"src.config.{args.config}")
        cfg = config_module.GPTConfig()
        if args.force_cpu:
            cfg.device = "cpu"
        if args.quantum:
            cfg.use_quantum = True
        elif args.classical:
            cfg.use_quantum = False
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    if args.mode == "compare":
        seeds = (
            [int(seed) for seed in args.seeds.split(",")] if args.seeds else [args.seed]
        )
        progress_path = os.path.join(
            "experiments", f"comparison_{args.name}.progress.json"
        )
        final_path = os.path.join("experiments", f"comparison_{args.name}.json")
        campaign = {
            "schema_version": "1.1",
            "experiment": "01_quantum_gpt",
            "status": "in_progress",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "dataset": args.dataset,
            "config_name": args.config,
            "seeds": seeds,
            "runs": {"classical": [], "quantum": []},
            "protocol": {
                "variant_order": ["classical", "quantum"],
                "shared_initialization": True,
                "split_specific_batch_generators": True,
                "selected_checkpoint_evaluation_seed_offset": 10_000,
            },
        }
        if args.resume:
            if not os.path.exists(progress_path):
                raise FileNotFoundError(f"No progress file found: {progress_path}")
            with open(progress_path, "r") as f:
                campaign = json.load(f)
            if (
                campaign["dataset"] != args.dataset
                or campaign["config_name"] != args.config
                or not set(campaign["seeds"]).issubset(seeds)
            ):
                raise ValueError(
                    "Resume arguments must keep dataset/config and may only "
                    "extend the existing seed list"
                )
            campaign["seeds"] = seeds
            campaign["status"] = "in_progress"
        results = campaign["runs"]

        def save_progress():
            save_json(progress_path, campaign)

        for seed in seeds:
            trainers = {}
            for variant, use_quantum in (("classical", False), ("quantum", True)):
                run_config = config_module.GPTConfig()
                run_config.use_quantum = use_quantum
                if args.force_cpu:
                    run_config.device = "cpu"
                set_seed(seed)
                trainers[variant] = Trainer(
                    run_config,
                    f"{args.name}_{variant}_s{seed}",
                    args.dataset,
                    seed=seed,
                    config_name=args.config,
                )
            shared = match_shared_initialization(
                trainers["classical"].model, trainers["quantum"].model
            )
            pairing = {
                "seed": seed,
                "shared_state_sha256": shared["sha256"],
                "matched_state_keys": shared["matched_keys"],
            }
            for trainer in trainers.values():
                trainer.pairing_metadata = pairing

            for variant in ("classical", "quantum"):
                if any(run["seed"] == seed for run in results[variant]):
                    print(f"Skipping completed {variant} seed {seed}")
                    continue
                run_dir = trainers[variant].train()
                with open(os.path.join(run_dir, "results.json"), "r") as f:
                    results[variant].append(json.load(f))
                results[variant].sort(key=lambda run: seeds.index(run["seed"]))
                save_progress()

        statistics = {}
        for metric in (
            "selected_val_loss",
            "selected_val_perplexity",
            "selected_loss_improvement_over_unigram",
            "total_training_time_seconds",
        ):
            statistics[metric] = paired_comparison(
                [run[metric] for run in results["quantum"]],
                [run[metric] for run in results["classical"]],
                seed=args.seed,
            )
        statistics["parameter_count"] = paired_comparison(
            [run["params"]["trainable"] for run in results["quantum"]],
            [run["params"]["trainable"] for run in results["classical"]],
            seed=args.seed,
        )
        campaign["paired_statistics"] = statistics
        campaign["status"] = "complete"
        campaign["completed_at"] = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )
        save_json(final_path, campaign)
        save_json(progress_path, campaign)
        return

    if args.mode in ["train", "full"]:
        set_seed(args.seed)
        trainer = Trainer(
            cfg,
            args.name,
            args.dataset,
            seed=args.seed,
            config_name=args.config,
        )
        try:
            args.run_dir = trainer.train()
        except BaseException as e:
            # Move partial logs so failed runs don't clutter experiments/
            failed_dir = os.path.join(
                "experiments_failed", os.path.basename(trainer.run_dir)
            )
            print(f"Training failed. Moving logs to {failed_dir}")
            shutil.move(trainer.run_dir, failed_dir)
            raise e

    if args.mode in ["generate", "full"]:
        if not args.run_dir:
            print("Error: --run_dir is required for generation mode.")
            return

        with open(os.path.join(args.run_dir, "config.json"), "r") as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                # Skip read-only properties (e.g. n_qubits is derived from n_embd/n_head)
                if not isinstance(getattr(type(cfg), key, None), property):
                    setattr(cfg, key, value)

        if args.force_cpu:
            cfg.device = "cpu"

        seeds = args.seeds.split(",") if args.seeds else []
        if not seeds:
            seeds = (
                [str(args.seed)]
                + [str(secrets.randbelow(1000000)) for _ in range(args.generations - 1)]
                if args.generations > 1
                else [args.seed]
            )

        gen = Generator(cfg, args.run_dir, args.dataset)
        results = gen.generate(
            hint=args.hint,
            seeds=seeds,
            max_new_tokens=args.tokens,
            print_in_place=args.print_in_place,
        )

        if not args.print_in_place:
            for i, res in enumerate(results):
                print(f"\n--- Result (Seed {seeds[i]}) ---\n{res}\n")


if __name__ == "__main__":
    main()
