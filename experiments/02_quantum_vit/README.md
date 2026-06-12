# Experiment 02 — Quantum Vision Transformer

A Vision Transformer for medical image classification where the Q/K/V
projections in self-attention are replaced by Variational Quantum Circuits.

**Result: falsified.** Equivalent accuracy to the classical ViT on MedMNIST,
with significantly higher training time. No measurable benefit from replacing
linear projections with VQC attention.

## Scientific concept

Classical ViT attention computes Q, K, V via linear projections of token embeddings.
Here each projection is a VQC with no classical bottleneck (embed_dim == n_qubits):

```
x (n_qubits,) → tanh·π → AngleEmbedding → StronglyEntanglingLayers → ⟨Z⟩ → y (n_qubits,)
```

The no-bottleneck constraint (learned from experiment 01) ensures the VQC operates
on the full embedding without lossy compression. Despite this, the Pauli-Z
expectation values produce outputs bounded in [-1, 1] that are functionally
similar to a tanh-activated linear layer.

## Architecture

```
28×28 image
    ↓
Patch Embedding  (Conv2d stride=patch_size → 16 patches × embed_dim)
    ↓
[CLS] + positional embedding  →  (17, embed_dim)
    ↓
┌──────────────────────────────────────────────┐  ×n_layer
│  LayerNorm → Quantum Attention → Residual    │
│  LayerNorm → FeedForward → Residual          │
└──────────────────────────────────────────────┘
    ↓
CLS token → LayerNorm → Linear(embed_dim, n_classes)
```

Per-head VQC triplet: Q-circuit, K-circuit, V-circuit (each `QuantumLinear`).

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
python run.py 02 compare --epochs 50

# Different dataset
python run.py 02 train --dataset bloodmnist --epochs 50
```

Options: `--dataset` (pathmnist|bloodmnist|dermamnist|fashionmnist|cifar10),
`--epochs`, `--embed-dim`, `--n-head`.

## Training outputs

Each run writes to `experiments/<run_name>/`:

| File | Content |
|------|---------|
| `config.json` | Full config snapshot |
| `best_model.pth` | Best checkpoint (highest val accuracy) |
| `final_model.pth` | Final epoch checkpoint |
| `results.json` | Test metrics and training history |
| `events.out.tfevents.*` | TensorBoard logs |

## Installation

All dependencies are managed from the repository root:

```bash
uv sync
```
