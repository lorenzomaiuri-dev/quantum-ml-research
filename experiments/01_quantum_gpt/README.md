# Experiment 01 — Quantum GPT

A character-level language model (inspired by Karpathy's nanoGPT) where the
Q/K/V projections in each attention head are replaced by Variational Quantum
Circuits (VQC).

**Result: falsified.** Comparable validation loss to the classical baseline,
with 9× more parameters and 7× slower training. The quantum bottleneck
(compressing n_embd → n_qubits per head) introduces lossy compression that
the VQC cannot compensate for.

## Scientific concept

In a standard Transformer, attention heads project inputs into Q, K, V spaces
via learned linear matrices. Here, each projection is a parameterized quantum
circuit:

```
x (n_embd) → Linear(n_embd, n_qubits) → tanh·π → AngleEmbedding
           → StronglyEntanglingLayers → ⟨Z⟩ → y (n_qubits)
```

The classical adapter (Linear) is the bottleneck: n_embd is typically much
larger than n_qubits (e.g. 32 → 4). The VQC operates in a 2^n_qubits Hilbert
space, but the information bottleneck prevents it from utilizing this capacity.

This is the key structural limitation identified by experiment 01, which
motivated the "no-bottleneck" design constraint adopted in experiments 02 and 03
(where embed_dim == n_qubits by design).

## Structure

```
01_quantum_gpt/
├── main.py                     # CLI: --mode train|generate|full
├── src/
│   ├── config/                 # Config variants: default, fast, light, big, heavy
│   │   └── base.py             # GPTConfig dataclass
│   ├── dataset.py              # Character-level tokenizer + batch sampler
│   ├── model.py                # QuantumGPT (Transformer with VQC attention)
│   ├── quantum_layers.py       # Re-exports QuantumLinearWithAdapter from quantum_framework
│   ├── training.py             # Trainer (step-based, text generation specific)
│   └── inference.py            # Generator (autoregressive sampling)
└── data/                       # Input text files (Shakespeare, Dante, etc.)
```

## Usage

From the repository root:

```bash
# Train on the default dataset (input.txt = Shakespeare)
python run.py 01 --mode train

# Train with a fast config (fewer layers/qubits, for quick testing)
python run.py 01 --mode train --config fast

# Generate text from a trained run
python run.py 01 --mode generate --run_dir experiments/01_quantum_gpt/experiments/<run_name>

# Train then generate in one command
python run.py 01 --mode full
```

Or directly from this directory:

```bash
python main.py --mode train --config fast
```

Available `--config` variants: `default`, `fast`, `light`, `big`, `heavy`.

## Training outputs

Each run writes to `experiments/<run_name>/`:

| File | Content |
|------|---------|
| `config.json` | Full config snapshot |
| `best_model.pth` | Best checkpoint (lowest val loss) |
| `final_model.pth` | Final epoch checkpoint |
| `metrics.json` | Loss history and training time |
| `dictionary.json` | Tokenizer vocabulary |
| `events.out.tfevents.*` | TensorBoard logs |

## Installation

All dependencies are managed from the repository root. Run once from there:

```bash
uv sync --extra gpt
```

The `[gpt]` extra adds `torchinfo`, `torchviz`, `datasets`, and `transformers`
which are used by this experiment's architecture visualization and tokenizer code.
