#!/usr/bin/env python
"""
Common entry point for all four quantum ML experiments.

Usage:
    python run.py                        # list all experiments
    python run.py <id> [args...]         # run an experiment, passing args through
    python run.py <id> --help            # show the experiment's own help

Examples:
    python run.py 01 --mode train --config fast
    python run.py 01 --mode generate --run_dir experiments/shakespeare_20260611_143022
    python run.py 02 train --epochs 50
    python run.py 02 train --classical --epochs 50
    python run.py 02 compare --epochs 50
    python run.py 03 train --model quantum_reg --dataset pathmnist --epochs 50
    python run.py 03 compare --dataset pathmnist --epochs 50
    python run.py 03 ablation --epochs 50
    python run.py 04

Each experiment runs from its own directory (experiments/0X_*/), so relative
paths to data/ and experiments/ subdirectories resolve correctly, exactly as
they would if you cd'd there manually.
"""

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent

EXPERIMENTS = {
    "01": {
        "dir": ROOT / "experiments" / "01_quantum_gpt",
        "description": "Quantum GPT — Shakespeare text generation with VQC attention",
        "synopsis": "python run.py 01 --mode [train|generate|full|compare] [--config NAME]",
    },
    "02": {
        "dir": ROOT / "experiments" / "02_quantum_vit",
        "description": "Quantum ViT — MedMNIST image classification with quantum patch embedding",
        "synopsis": "python run.py 02 [train|compare] [--epochs N] [--classical] [--seeds ...]",
    },
    "03": {
        "dir": ROOT / "experiments" / "03_quantum_reg",
        "description": "Quantum Regularization — ablation: vanilla vs bounded_mlp vs quantum_reg",
        "synopsis": "python run.py 03 [train|compare|ablation] [--model MODEL] [--dataset DATASET]",
    },
    "04": {
        "dir": ROOT / "experiments" / "04_quantum_kernel",
        "description": "Quantum Kernel PoC — quantum vs classical similarity on BreastMNIST (~5 min)",
        "synopsis": "python run.py 04 [--repeats N] [--seed N]",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_help() -> None:
    print(__doc__)
    print("Available experiments:\n")
    for eid, info in EXPERIMENTS.items():
        print(f"  {eid}  {info['description']}")
        print(f"      {info['synopsis']}")
        print()


def run_experiment(eid: str, extra_args: list[str]) -> int:
    """
    Launch experiments/0X_*/main.py in its own directory.

    Uses the same Python interpreter that is running this script, so
    `uv run python run.py ...` correctly picks up the uv-managed venv.
    """
    info = EXPERIMENTS[eid]
    exp_dir = info["dir"]

    if not exp_dir.is_dir():
        print(f"Error: experiment directory not found: {exp_dir}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "main.py", *extra_args]
    print(f"[run.py] cd {exp_dir.relative_to(ROOT)}", flush=True)
    print(f"[run.py] {' '.join(cmd)}\n", flush=True)

    result = subprocess.run(cmd, cwd=exp_dir)
    return result.returncode


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = sys.argv[1:]

    if not args:
        print_help()
        return 0

    eid = args[0]

    # Normalise: accept "1", "01", "1_quantum_gpt" etc.
    for key in EXPERIMENTS:
        if eid == key or eid == key.lstrip("0") or eid.startswith(key):
            return run_experiment(key, args[1:])

    print(
        f"Error: unknown experiment '{eid}'. Choose from: {', '.join(EXPERIMENTS)}",
        file=sys.stderr,
    )
    print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
