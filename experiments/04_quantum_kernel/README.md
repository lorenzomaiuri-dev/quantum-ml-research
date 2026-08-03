# Experiment 04 — Quantum Kernel PoC

Does an IQP fidelity kernel (Havlíček et al. 2019) separate medical image
classes better than classical cosine and RBF kernels?

**Result: rejected in the opposite direction on the primary metric.** Across 20
paired subsamples, the IQP Fisher score is lower than cosine by 0.0152 (95% CI
[-0.0238, -0.0065]) and lower than RBF by 0.0417 (95% CI
[-0.0644, -0.0190]). Held-out kernel-SVC validation also favours both classical
baselines across three class-sampling regimes. These findings apply to this
8-component feature map on BreastMNIST, not to quantum kernels in general.

## Method

1. Load BreastMNIST (`0=malignant`, `1=normal/benign`).
2. Sample 80 class-0 and 20 class-1 images without replacement using a recorded
   seed. This inverts the natural prevalence and is therefore countervalidated
   with 50/50 and 27/73 samples.
3. Compress each 28×28 image to 8 dimensions via deterministic full-SVD PCA
   (fixed, non-trainable).
4. Scale PCA features to [0, π] for angle embedding.
5. Compute three kernel matrices:
   - **Cosine** on the angle-scaled vectors in ℝ⁸;
   - **RBF** on centered PCA features, with label-free median bandwidth;
   - **IQP fidelity**: |⟨φ(x)|φ(y)⟩|² in 2⁸ dimensions, using two
     `qml.IQPEmbedding` repetitions and adjacent ZZ feature-product phases.
6. Measure class separation via Fisher discriminant and intra/inter-class ratio.

No training, no gradients — pure linear algebra + quantum circuits.

## Structure

```
04_quantum_kernel/
├── main.py                    # Separation campaign and kernel matrices
└── classifier_validation.py   # Training-only tuning, official-test SVC metrics
```

## Usage

```bash
# Primary 20-repeat campaign from the repository root
python run.py 04 --feature-map iqp --iqp-repeats 2 \
  --kernel-method pairwise --repeats 20 --seed 42

# Exact statevector cross-check (simulator only)
python run.py 04 --feature-map iqp --kernel-method statevector \
  --repeats 20 --seed 42

# Held-out classifier validation
cd experiments/04_quantum_kernel
python classifier_validation.py --n-class0 50 --n-class1 50 \
  --repeats 20 --seed 42
```

Ogni campagna salva `campaign_results.json`; ogni ripetizione salva
`results.json`, `run_manifest.json` e `kernel_matrices.npz`. Le statistiche
appaiate usano la differenza quantum-minus-classical sullo stesso sottocampione.

L'evidenza compatta definitiva, comprensiva dei tre regimi, della validazione
SVC e del controllo statevector, è congelata in
`docs/checkpoints/experiment_04_quantum_kernel_frozen.json`. Per rigenerarla e
produrre le figure della tesi usare `analyze_results.py`; il comando completo è
documentato in `docs/runbook.md`.

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

About 38 seconds per IQP pairwise repeat on the reference CPU. Runtime depends
on the PennyLane version and processor; the bottleneck is computing the 100×100
quantum kernel matrix (5050 unique circuit evaluations, each an 8-qubit
simulation). The exact statevector cross-check takes about 0.4 seconds per
repeat but is not an execution strategy available on quantum hardware.

## Historical feature map

`--feature-map custom_cnot_rz` preserves the repository's original
H/RZ/CNOT/RZ circuit for auditability. It is a valid fidelity feature map but
is not an IQP embedding: the earlier documentation incorrectly attributed it
to Havlíček et al. The thesis result uses `--feature-map iqp`.

## Installation

All dependencies are managed from the repository root:

```bash
uv sync
```
