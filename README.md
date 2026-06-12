# Quantum ML Research

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-red)
![PennyLane](https://img.shields.io/badge/PennyLane-0.43%2B-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Four experiments investigating whether Variational Quantum Circuits (VQC)
offer measurable advantages over classical counterparts in machine learning tasks.
All experiments run on classical quantum simulators; results reflect expressivity
and regularization properties, not computational speedup.

**All four hypotheses were falsified.**

## Experiments

| # | Hypothesis | Result |
|---|-----------|--------|
| [01 — Quantum GPT](experiments/01_quantum_gpt/) | VQC Q/K/V projections improve GPT perplexity | Comparable loss, 9× more parameters, 7× slower |
| [02 — Quantum ViT](experiments/02_quantum_vit/) | VQC patch embedding improves ViT classification | Equivalent accuracy on MedMNIST, significantly slower |
| [03 — Quantum Reg](experiments/03_quantum_reg/) | VQC acts as implicit geometric regularizer | No statistically significant generalization gap reduction |
| [04 — Quantum Kernel](experiments/04_quantum_kernel/) | IQP quantum kernel improves class separability | Classical cosine kernel equivalent or better on BreastMNIST |

Full methodology and structural limitations in [`docs/research_summary.md`](docs/research_summary.md).

## Repository structure

```
quantum-ml-research/
├── run.py                       # Common entry point for all experiments
├── pyproject.toml               # Single dependency manifest (uv)
├── uv.lock                      # Pinned environment
├── requirements.txt             # Human-readable dependency summary
├── LICENSE
│
├── quantum_framework/           # Shared library (installed as a package)
│   ├── layers/
│   │   └── quantum_linear.py   # QuantumLinear, QuantumLinearWithAdapter
│   ├── data/
│   │   └── medmnist_loader.py  # MedMNIST loading, noise corruption
│   ├── training/
│   │   └── trainer.py          # Generic image-classification training loop
│   ├── evaluation/
│   │   └── metrics.py          # Accuracy, AUC, F1, generalization gap
│   └── utils/
│       ├── config.py           # BaseViTConfig dataclass
│       └── experiment.py       # set_seed, make_run_dir, save_json
│
├── experiments/
│   ├── 01_quantum_gpt/         # Hybrid QNN-NanoGPT (text generation)
│   ├── 02_quantum_vit/         # Quantum Vision Transformer (MedMNIST)
│   ├── 03_quantum_reg/         # Ablation: quantum as implicit regularizer
│   └── 04_quantum_kernel/      # PoC: quantum kernel vs cosine similarity
│
└── docs/
    └── research_summary.md     # Full research summary and results
```

Each experiment directory has its own `README.md`, `main.py`, and `src/` package.
The experiment-specific files are thin wrappers around `quantum_framework`;
research code (model architectures, VQC circuits) is not shared.

## Installation

This repository uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone and enter the repository
git clone https://github.com/lorenzomaiuri-dev/quantum-ml-research.git
cd quantum-ml-research

# Create virtual environment and install all dependencies
uv sync

# For experiment 01 (requires torchinfo, torchviz, transformers)
uv sync --extra gpt

# For GPU-accelerated PennyLane simulation (requires CUDA)
uv sync --extra gpu
```

After `uv sync`, activate the environment once if you prefer calling `python` directly:

```bash
source .venv/bin/activate
```

Or prefix every command with `uv run`:

```bash
uv run python run.py 02 train --epochs 50
```

## Running experiments

All experiments can be launched from the repository root via `run.py`:

```bash
python run.py                                       # list all experiments

# Experiment 01 — Quantum GPT
python run.py 01 --mode train
python run.py 01 --mode train --config fast
python run.py 01 --mode generate --run_dir experiments/01_quantum_gpt/experiments/<run_name>
python run.py 01 --mode full

# Experiment 02 — Quantum ViT
python run.py 02 train --epochs 50
python run.py 02 train --classical --epochs 50      # classical baseline
python run.py 02 compare --epochs 50               # both side by side

# Experiment 03 — Quantum Regularization
python run.py 03 train --model quantum_reg --dataset pathmnist --epochs 50
python run.py 03 compare --dataset pathmnist --epochs 50
python run.py 03 ablation --epochs 50              # full 3×5×3 = 45 runs

# Experiment 04 — Quantum Kernel (~5 min, no training)
python run.py 04
```

Alternatively, cd into each experiment and run `python main.py` directly —
the CLI is identical.

## quantum_framework

`quantum_framework` is a Python package installed in editable mode by `uv sync`.
It provides the infrastructure shared across experiments, without touching
research code (model architectures remain experiment-local).

```python
from quantum_framework.layers import QuantumLinear, QuantumLinearWithAdapter
from quantum_framework.data import load_medmnist, make_noisy_loader
from quantum_framework.training import BaseTrainer
from quantum_framework.evaluation import compute_classification_metrics, generalization_gap
from quantum_framework.utils import BaseViTConfig, set_seed, make_run_dir
```

The VQC pipeline used in all experiments:

```
input → tanh(x)·π  →  AngleEmbedding  →  StronglyEntanglingLayers  →  ⟨Z_i⟩  →  output
```

Weight shape: `(n_qlayers, n_qubits, 3)`. Differentiation via `diff_method="backprop"`.

## GPU acceleration

By default, all experiments use `q_device="default.qubit"` (CPU simulation).
For GPU-accelerated simulation, install `pennylane-lightning-gpu` and change
the device string:

```bash
uv sync --extra gpu
```

```python
config.q_device = "lightning.qubit"
```

## License

MIT — see [LICENSE](LICENSE).
