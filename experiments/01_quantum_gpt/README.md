# Experiment 01 — Quantum GPT

A character-level language model based on the didactic Transformer from
Karpathy's *Neural Networks: Zero to Hero* course, with nanoGPT as an additional
design reference. The Q/K/V projections in each attention head are replaced by
Variational Quantum Circuits (VQC).

**Status: definitive campaign complete.** Ten paired runs with the preregistered
`thesis` preset found no detectable validation-loss or perplexity advantage.
The mean quantum-minus-classical perplexity difference is +0.067 (95% CI
[-0.141, +0.275], paired t-test p=0.484). Both variants clearly beat the
held-out unigram baseline, while the quantum simulator is 46.5x slower and the
quantum model has 11.4% more parameters. The frozen aggregate is
`docs/checkpoints/experiment_01_quantum_gpt_n10.json`.

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

This is the structural limitation tested by experiment 01 and motivates the
"no-bottleneck" design constraint adopted in experiments 02 and 03 (where
`embed_dim == n_qubits` by design). Whether it harms predictive quality is an
empirical question answered by the paired campaign rather than assumed here.

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

# Paired variants over identical seeds and configuration
python run.py 01 --mode compare --config fast --seeds 42,137,256,512,1024

# Definitive thesis campaign; add --resume after an interruption
python run.py 01 --mode compare --config thesis --dataset data/input.txt \
  --seeds 42,137,256,512,1024,2048,4096,8192,16384,32768 \
  --name thesis_shakespeare --force_cpu --resume

# Generate text from a trained run
python run.py 01 --mode generate --run_dir experiments/01_quantum_gpt/experiments/<run_name>

# Train then generate in one command
python run.py 01 --mode full
```

Or directly from this directory:

```bash
python main.py --mode train --config fast
```

Available `--config` variants: `default`, `fast`, `light`, `thesis`, `big`,
`heavy`. The `thesis` preset uses character tokens, a 16-token context and 1500
iterations; it was selected before the definitive campaign because its pilot
clearly beat the held-out unigram baseline.

## Reproduce the figures

```bash
python experiments/01_quantum_gpt/analyze_results.py \
  docs/checkpoints/experiment_01_quantum_gpt_n10.json \
  --output-dir docs/checkpoints/figures
```

## Training outputs

Each run writes to `experiments/<run_name>/`:

| File | Content |
|------|---------|
| `config.json` | Full config snapshot |
| `run_manifest.json` | Commit, comando, seed, ambiente e versioni |
| `params.json` | Parametri totali e addestrabili |
| `best_model.pth` | Best checkpoint (lowest val loss) |
| `final_model.pth` | Final epoch checkpoint |
| `metrics.json`, `results.json` | Loss, perplexity, parametri e tempi |
| `dictionary.json` | Tokenizer vocabulary |
| `events.out.tfevents.*` | TensorBoard logs |

## Installation

All dependencies are managed from the repository root. Run once from there:

```bash
uv sync --extra gpt
```

The `[gpt]` extra adds `torchinfo`, `torchviz`, `datasets`, and `transformers`
which are used by this experiment's architecture visualization and tokenizer code.
