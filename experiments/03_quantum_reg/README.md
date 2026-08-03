# Experiment 03 — Quantum Regularization Ablation Study

An ablation study investigating whether VQC patch embeddings act as implicit
regularizers in hybrid quantum-classical Vision Transformers.

**Status: definitive paired campaign complete (45/45 runs).** The hypothesis is
not supported: the quantum-minus-bounded mean generalization gap is positive
on PathMNIST (+0.148), BloodMNIST (+0.100), and DermaMNIST (+0.085). Earlier
unpaired runs remain exploratory and are excluded from this conclusion.

## Hypothesis

Unitary constraints and entanglement impose geometric regularization on the
optimization landscape, reducing overfitting compared to a classical MLP with
equal or greater parameter count.

## Experimental design

Three models share an **identical** ViT body. They differ only in the patch
embedding layer:

| Model | Embedding | Bounded | Unitary | Params |
|-------|-----------|---------|---------|--------|
| **A. Vanilla** | `Linear(patch_dim, embed_dim)` | No | No | 1,470 |
| **B. Bounded MLP** | `Linear → tanh·π → MLP(10→20→10) → tanh` | Yes | No | 1,870 |
| **C. Quantum-Reg** | `Linear → tanh·π → VQC → ⟨Z⟩` | Yes | Yes | 1,530 |

**Key comparison: B vs C.** Same bounded I/O, same compression layer, but C
adds unitarity and entanglement. Model B has 340 more parameters. Any lower
generalization gap for C is attributable to quantum geometry alone.

Primary metric: **generalization gap** = train_acc − test_acc.
A valid result requires gap(C) < gap(B) consistently across 3+ datasets and
5+ seeds with p < 0.05.

The observed result fails this criterion in the opposite direction. On
PathMNIST the higher quantum gap has an unadjusted paired `p=0.026`; the other
two dataset-level intervals include zero. The appropriate conclusion is no
evidence of quantum regularization, not a universal anti-regularization claim.

## Structure

```
03_quantum_reg/
├── main.py                         # CLI: train | compare | ablation
├── src/
│   ├── config.py                   # AblationConfig (extends BaseViTConfig)
│   ├── data/
│   │   └── loaders.py              # Thin wrapper → quantum_framework.data
│   ├── models/
│   │   ├── patch_embeddings.py     # Models A, B, C + factory function
│   │   └── vit.py                  # Shared ViT body (identical for all 3)
│   └── engine/
│       └── trainer.py              # Thin wrapper → quantum_framework.training.BaseTrainer
└── experiments/                    # Auto-generated run directories (gitignored)
```

## Usage

From the repository root:

```bash
# Train a single model
python run.py 03 train --model quantum_reg --dataset pathmnist --epochs 50

# Compare all 3 models (single seed)
python run.py 03 compare --dataset pathmnist --epochs 50

# Definitive ablation: 3 models × 5 seeds × 3 datasets = 45 runs
python run.py 03 ablation --epochs 400 --train-subset 100 \
  --eval-interval 40 --dropout 0 --weight-decay 0 \
  --name thesis_regularization_n100_steps400 --skip-noise

# Resume an interrupted campaign, including an interrupted epoch
python run.py 03 ablation --epochs 400 --train-subset 100 \
  --eval-interval 40 --dropout 0 --weight-decay 0 \
  --name thesis_regularization_n100_steps400 --skip-noise --resume

# Scarce-data regime (1000 training samples, stratified)
python run.py 03 compare --dataset pathmnist --epochs 50 --train-subset 1000
```

`--model` choices: `vanilla`, `bounded_mlp`, `quantum_reg`.
`--dataset` choices: `pathmnist`, `bloodmnist`, `dermamnist`.

The primary campaign omits the ten noise-corruption evaluations per run. They
can be performed as a secondary checkpoint analysis if the primary gap result
warrants it; this avoids multiplying simulator time before a finding exists.
The scarce-data regime was selected using only bounded-MLP train/validation
metrics at a fixed 400-step budget; no quantum result or test metric was used.

L'aggregato finale include valori grezzi, riepiloghi e confronti appaiati
`quantum_reg`–`bounded_mlp` e `quantum_reg`–`vanilla`: intervallo di confidenza,
bootstrap, paired t-test, Wilcoxon e Cohen's $d_z$.

To reproduce the artifact audit, metric countervalidation, and thesis figures:

```bash
python experiments/03_quantum_reg/validate_results.py \
  experiments/03_quantum_reg/experiments/ablation_thesis_regularization_n100_steps400.json \
  --output docs/checkpoints/experiment_03_checkpoint_validation_n100_steps400.json

python experiments/03_quantum_reg/analyze_results.py \
  experiments/03_quantum_reg/experiments/ablation_thesis_regularization_n100_steps400.json \
  --output-dir docs/checkpoints/figures
```

## Training outputs

Each run writes to `experiments/<run_name>/`:

| File | Content |
|------|---------|
| `config.json` | Full config snapshot |
| `params.json` | Parameter counts by component |
| `best_model.pth` | Best checkpoint (highest val accuracy) |
| `final_model.pth` | Final epoch checkpoint |
| `results.json` | Full results: gen gap, noise robustness, per-epoch history |
| `events.out.tfevents.*` | TensorBoard: loss, accuracy, gap, grad norms, saturation |

## Interpreting results

A valid quantum regularization finding requires all of:

1. `train_acc(B) ≈ train_acc(C)` — both models learn comparably
2. `gen_gap(C) < gen_gap(B)` — quantum generalizes better
3. Consistent across 3+ datasets and 5+ seeds
4. p < 0.05 (paired t-test across seeds)

If C shows lower train_acc than B, the VQC is underfitting. Check
`act_saturation` in results: high saturation (> 0.3) means tanh·π is killing
gradients through the circuit.

## Installation

All dependencies are managed from the repository root:

```bash
uv sync
```
