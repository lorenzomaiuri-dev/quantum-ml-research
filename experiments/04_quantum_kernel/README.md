# Experiment 04 — Quantum Kernel PoC

Does a quantum kernel (IQP-style embedding, Havlicek et al. 2019) separate
medical image classes better than classical cosine similarity?

**Result: falsified.** The classical cosine kernel achieves equivalent or better
class separability. The quantum Hilbert space provides no measurable advantage
for this low-dimensional embedding.

## Method

1. Load BreastMNIST (binary: normal vs malignant, heavily imbalanced).
2. Sample 80 class-0 and 20 class-1 images without replacement using a recorded seed.
3. Compress each 28×28 image to 8 dimensions via PCA (fixed, non-trainable).
4. Scale PCA features to [0, π] for angle embedding.
5. Compute two similarity matrices:
   - **Classical**: cosine similarity in ℝ⁸
   - **Quantum**: |⟨φ(x)|φ(y)⟩|² in 2⁸ = 256-dimensional Hilbert space
6. Measure class separation via Fisher discriminant and intra/inter-class ratio.

No training, no gradients — pure linear algebra + quantum circuits.

## Structure

```
04_quantum_kernel/
└── main.py     # Self-contained: data loading, kernel computation, analysis
```

## Usage

```bash
# From repository root (~5 minutes)
python run.py 04

# Repeated subsampling for paired uncertainty estimates
python run.py 04 --repeats 5 --seed 42

# Or directly
cd experiments/04_quantum_kernel
python main.py
```

Ogni campagna salva `campaign_results.json`; ogni ripetizione salva
`results.json`, `run_manifest.json` e `kernel_matrices.npz`. Le statistiche
appaiate usano la differenza quantum-minus-classical sullo stesso sottocampione.

Expected output:

```
VERDICT
══════════════════════════════════════════════════
  Fisher discriminant: Classical=X.XXXX | Quantum=X.XXXX
  Separation ratio:    Classical=X.XXXX | Quantum=X.XXXX
  ✗ CLASSICAL WINS  (or ≈ NO SIGNIFICANT DIFFERENCE)
══════════════════════════════════════════════════
```

## Expected runtime

~5 minutes on CPU. The bottleneck is computing the 100×100 quantum kernel
matrix (5050 unique circuit evaluations, each an 8-qubit simulation).

## Installation

All dependencies are managed from the repository root:

```bash
uv sync
```
