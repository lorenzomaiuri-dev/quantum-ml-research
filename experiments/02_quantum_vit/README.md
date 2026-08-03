# Experiment 02 — Quantum Vision Transformer

A Vision Transformer for medical image classification where the linear patch
embedding is replaced by a Variational Quantum Circuit.

**Status: definitive paired campaign in progress.** Historical runs are either
unpaired or incomplete and are not evidence for or against the hypothesis. The
official PathMNIST campaign uses five paired seeds, identical shared
initialisation, data order and dropout RNG stream, best-validation model
selection, and crash-safe epoch checkpoints.

## Scientific concept

Each flattened patch is first compressed from `patch_dim` to `embed_dim`; the
compressed features are then angle-encoded in a VQC (`embed_dim == n_qubits`):

```
x (patch_dim,) → Linear → tanh·π → AngleEmbedding → StronglyEntanglingLayers → ⟨Z⟩
```

The classical baseline uses `Linear(patch_dim, embed_dim)` at the same location.
The Transformer encoder that follows is classical in both variants.

## Architecture

```
28×28 image
    ↓
Patch extraction → Linear compression → optional VQC
    ↓
[CLS] + positional embedding  →  (17, embed_dim)
    ↓
┌──────────────────────────────────────────────┐  ×n_layer
│  LayerNorm → Classical Attention → Residual  │
│  LayerNorm → FeedForward → Residual          │
└──────────────────────────────────────────────┘
    ↓
CLS token → LayerNorm → Linear(embed_dim, n_classes)
```

`src/model.py` contains an earlier quantum-attention prototype, but the official
CLI in `main.py` uses `src/engine/trainer.py` and `HybridQCNNViT`.

## Structure

```
02_quantum_vit/
├── main.py                     # CLI: train | compare
├── src/
│   ├── config.py               # QViTConfig (extends BaseViTConfig)
│   ├── dataset.py              # Thin wrapper → quantum_framework.data
│   ├── model.py                # QuantumViT with quantum/classical attention
│   ├── quantum_layers.py       # Re-exports QuantumLinear from quantum_framework
│   ├── trainer.py              # Thin wrapper → quantum_framework.training.BaseTrainer
│   ├── data/
│   │   ├── base.py             # Dispatcher: MedMNIST or torchvision
│   │   ├── medmnist.py         # Thin wrapper → quantum_framework.data
│   │   └── torchvision.py      # Loader for CIFAR-10, FashionMNIST
│   ├── engine/
│   │   ├── metrics.py          # Re-exports compute_classification_metrics
│   │   └── trainer.py          # Thin wrapper (HybridQCNNViT variant)
│   └── models/
│       ├── hybrid_vit.py       # HybridQCNNViT (quantum patch embedding variant)
│       ├── baselines.py        # Classical baseline models
│       └── layers/
│           ├── qcnn.py         # QuantumPatchEmbedding (QCNN variant)
│           └── transformer.py  # Classical transformer building blocks
└── experiments/                # Auto-generated run directories (gitignored)
```

## Usage

From the repository root:

```bash
# Train quantum model on PathMNIST
python run.py 02 train --epochs 50

# Train classical baseline
python run.py 02 train --classical --epochs 50

# Train both and compare side by side
python run.py 02 compare --dataset pathmnist --epochs 50 \
  --seeds 42 137 256 512 1024 --name thesis_pathmnist \
  --q-device default.qubit --force-cpu

# Resume the same campaign after an interruption
python run.py 02 compare --dataset pathmnist --epochs 50 \
  --seeds 42 137 256 512 1024 --name thesis_pathmnist \
  --q-device default.qubit --force-cpu --resume

# Different dataset
python run.py 02 train --dataset bloodmnist --epochs 50
```

Options: `--dataset` (pathmnist|bloodmnist|dermamnist|fashionmnist|cifar10),
`--epochs`, `--embed-dim`, `--n-head`, `--batch-size`, `--train-subset`,
`--q-device`, `--seed`, `--seeds`, `--name`, `--resume`, `--force-cpu`.

## Training outputs

Each run writes to `experiments/<run_name>/`:

| File | Content |
|------|---------|
| `config.json` | Full config snapshot |
| `run_manifest.json` | Commit, comando, ambiente e versioni |
| `params.json` | Parametri totali e addestrabili |
| `best_model.pth` | Best checkpoint (highest val accuracy) |
| `latest_checkpoint.pth` | Optimiser, scheduler and RNG state for exact epoch-level resume |
| `final_model.pth` | Final epoch checkpoint |
| `results.json` | Test metrics and training history |
| `events.out.tfevents.*` | TensorBoard logs |

Il comando `compare` salva inoltre `comparison_<dataset>.json`, con run grezze
e statistiche appaiate per accuracy, AUC, F1, gap di generalizzazione e tempo.

## Installation

All dependencies are managed from the repository root:

```bash
uv sync
```
