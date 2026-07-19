import argparse
import os
import time

import medmnist
import numpy as np
import pennylane as qml
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from quantum_framework.evaluation import paired_comparison
from quantum_framework.utils import collect_run_metadata, make_run_dir, save_json


def load_samples(dataset_name, n_class0, n_class1, n_qubits, seed=42):
    """Load and PCA-compress samples from two classes."""
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split="train", download=True, size=28)

    # Separate by class
    images_0, images_1 = [], []
    indices_0, indices_1 = [], []
    for index, (img, label) in enumerate(ds):
        img_flat = np.array(img).flatten() / 255.0
        lab = int(label.squeeze())
        if lab == 0:
            images_0.append(img_flat)
            indices_0.append(index)
        elif lab == 1:
            images_1.append(img_flat)
            indices_1.append(index)

    if len(images_0) < n_class0 or len(images_1) < n_class1:
        raise ValueError("requested samples exceed the available examples in a class")
    rng = np.random.default_rng(seed)
    selected_0 = rng.choice(len(images_0), size=n_class0, replace=False)
    selected_1 = rng.choice(len(images_1), size=n_class1, replace=False)
    images_0 = np.asarray(images_0)[selected_0]
    images_1 = np.asarray(images_1)[selected_1]
    sample_indices = {
        "class_0": [indices_0[i] for i in selected_0],
        "class_1": [indices_1[i] for i in selected_1],
    }

    print(f"Loaded: {len(images_0)} class-0, {len(images_1)} class-1")
    print(f"Raw dim: {images_0.shape[1]}")

    # PCA compression to n_qubits dimensions
    all_images = np.vstack([images_0, images_1])
    scaler = StandardScaler()
    all_images = scaler.fit_transform(all_images)

    pca = PCA(n_components=n_qubits)
    all_compressed = pca.fit_transform(all_images)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA → {n_qubits} dims (variance explained: {explained:.3f})")

    # Scale to [0, π] for angle embedding
    scaler2 = MinMaxScaler(feature_range=(0, np.pi))
    all_scaled = scaler2.fit_transform(all_compressed)

    x0 = all_scaled[: len(images_0)]
    x1 = all_scaled[len(images_0) :]

    return (
        x0,
        x1,
        {
            "sample_indices": sample_indices,
            "pca_explained_variance": float(explained),
        },
    )


def classical_similarity(x0, x1):
    """Compute dot-product similarity matrix."""
    all_x = np.vstack([x0, x1])
    # Normalize for cosine similarity
    norms = np.linalg.norm(all_x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_normed = all_x / norms
    sim = all_normed @ all_normed.T
    return sim


def quantum_kernel_matrix(x0, x1, n_qubits, q_device="default.qubit"):
    """
    Compute quantum kernel matrix using IQP-style embedding.

    K(x, y) = |⟨0|U†(y)U(x)|0⟩|²

    where U(x) = H⊗n · diag(exp(i·x)) · CNOT_cascade · H⊗n · diag(exp(i·x))

    This is a standard kernel from the QML literature (Havlicek et al., Nature 2019).
    No trainable parameters — purely a function of the data.
    """
    dev = qml.device(q_device, wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x, y):
        # Encode x
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)

        # Encode x† (adjoint = reverse)
        for i in range(n_qubits):
            qml.RZ(-y[i], wires=i)
        for i in range(n_qubits - 2, -1, -1):
            qml.CNOT(wires=[i, i + 1])
        for i in range(n_qubits):
            qml.RZ(-y[i], wires=i)
            qml.Hadamard(wires=i)

        # Probability of measuring |0...0⟩
        return qml.probs(wires=range(n_qubits))

    all_x = np.vstack([x0, x1])
    n = len(all_x)
    K = np.zeros((n, n))

    total_pairs = n * (n + 1) // 2
    computed = 0

    print(f"Computing quantum kernel ({n}×{n} = {total_pairs} unique pairs)...")
    start = time.time()

    for i in range(n):
        for j in range(i, n):
            probs = kernel_circuit(all_x[i], all_x[j])
            k_val = probs[0]  # |⟨0...0|U†(y)U(x)|0...0⟩|²
            K[i, j] = k_val
            K[j, i] = k_val
            computed += 1

            if computed % 500 == 0:
                elapsed = time.time() - start
                eta = elapsed / computed * (total_pairs - computed)
                print(
                    f"  {computed}/{total_pairs} pairs ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)"
                )

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    return K, elapsed


def analyze_separation(sim_matrix, n0, n1, label=""):
    """
    Measure class separation quality from a similarity matrix.

    Metrics:
    - intra_0:  mean similarity within class 0
    - intra_1:  mean similarity within class 1
    - inter:    mean similarity between classes
    - ratio:    (intra_0 + intra_1) / (2 * inter)
               Higher = better separation

    - fisher:   Fisher's criterion = (μ_intra - μ_inter)² / (σ_intra² + σ_inter²)
               Higher = more discriminative
    """
    # Extract intra- and inter-class blocks
    block_00 = sim_matrix[:n0, :n0]
    block_11 = sim_matrix[n0 : n0 + n1, n0 : n0 + n1]
    block_01 = sim_matrix[:n0, n0 : n0 + n1]

    # Remove diagonal from intra-class
    intra_0_vals = block_00[np.triu_indices_from(block_00, k=1)]
    intra_1_vals = block_11[np.triu_indices_from(block_11, k=1)]
    inter_vals = block_01.flatten()

    intra_0 = np.mean(intra_0_vals)
    intra_1 = np.mean(intra_1_vals)
    inter = np.mean(inter_vals)

    intra_mean = (intra_0 + intra_1) / 2
    ratio = intra_mean / inter if inter > 1e-10 else float("inf")

    # Fisher's discriminant
    intra_all = np.concatenate([intra_0_vals, intra_1_vals])
    fisher_num = (np.mean(intra_all) - np.mean(inter_vals)) ** 2
    fisher_den = np.var(intra_all) + np.var(inter_vals)
    fisher = fisher_num / fisher_den if fisher_den > 1e-10 else 0

    print(f"\n  {label}")
    print(f"  {'─' * 40}")
    print(f"  Intra-class 0 (normal):   {intra_0:.4f}")
    print(f"  Intra-class 1 (malign):   {intra_1:.4f}")
    print(f"  Inter-class:              {inter:.4f}")
    print(f"  Separation ratio:         {ratio:.4f}  (>1 = classes cluster)")
    print(f"  Fisher discriminant:      {fisher:.4f}  (higher = better)")

    return {
        "intra_0": float(intra_0),
        "intra_1": float(intra_1),
        "inter": float(inter),
        "ratio": float(ratio),
        "fisher": float(fisher),
    }


def run_once(args, seed, run_dir):
    """Execute and persist one paired classical/quantum kernel comparison."""
    run_start = time.time()

    print("=" * 50)
    print("  QUANTUM KERNEL PoC")
    print("  Does quantum similarity separate classes better")
    print("  than classical dot product?")
    print("=" * 50)

    # Load data
    print("\n--- Loading data ---")
    x0, x1, preprocessing = load_samples(
        args.dataset, args.n_class0, args.n_class1, args.n_qubits, seed
    )

    # Classical similarity
    print("\n--- Classical Kernel (cosine similarity) ---")
    classical_sim = classical_similarity(x0, x1)
    classical_metrics = analyze_separation(classical_sim, len(x0), len(x1), "CLASSICAL")

    # Quantum kernel
    print("\n--- Quantum Kernel (IQP embedding) ---")
    quantum_sim, quantum_time = quantum_kernel_matrix(
        x0, x1, args.n_qubits, args.q_device
    )
    quantum_metrics = analyze_separation(quantum_sim, len(x0), len(x1), "QUANTUM")

    # Verdict
    print(f"\n{'=' * 50}")
    print("  VERDICT")
    print(f"{'=' * 50}")

    q_fisher = quantum_metrics["fisher"]
    c_fisher = classical_metrics["fisher"]
    q_ratio = quantum_metrics["ratio"]
    c_ratio = classical_metrics["ratio"]

    print(f"  Fisher discriminant: Classical={c_fisher:.4f} | Quantum={q_fisher:.4f}")
    print(f"  Separation ratio:   Classical={c_ratio:.4f} | Quantum={q_ratio:.4f}")

    if q_fisher > c_fisher * 1.1:
        verdict = "quantum_higher"
    elif c_fisher > q_fisher * 1.1:
        verdict = "classical_higher"
    else:
        verdict = "within_10_percent"
    print(f"  Descriptive verdict: {verdict}")

    result = {
        "schema_version": "1.0",
        "experiment": "04_quantum_kernel",
        "variant": "iqp_fidelity_vs_cosine",
        "dataset": args.dataset,
        "seed": seed,
        "n_qubits": args.n_qubits,
        "n_class0": args.n_class0,
        "n_class1": args.n_class1,
        "q_device": args.q_device,
        "preprocessing": preprocessing,
        "classical": classical_metrics,
        "quantum": quantum_metrics,
        "differences": {
            "fisher": q_fisher - c_fisher,
            "ratio": q_ratio - c_ratio,
        },
        "descriptive_verdict": verdict,
        "quantum_kernel_time_seconds": quantum_time,
        "total_wall_time_seconds": time.time() - run_start,
    }
    save_json(os.path.join(run_dir, "results.json"), result)
    save_json(
        os.path.join(run_dir, "run_manifest.json"),
        collect_run_metadata("04_quantum_kernel", seed, "iqp_fidelity_vs_cosine"),
    )
    np.savez_compressed(
        os.path.join(run_dir, "kernel_matrices.npz"),
        classical=classical_sim,
        quantum=quantum_sim,
        x0=x0,
        x1=x1,
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Quantum kernel versus cosine similarity"
    )
    parser.add_argument("--dataset", default="breastmnist")
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-class0", type=int, default=80)
    parser.add_argument("--n-class1", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--q-device", default="default.qubit")
    args = parser.parse_args()
    if min(args.n_qubits, args.n_class0, args.n_class1, args.repeats) <= 0:
        parser.error("sample counts, qubits and repeats must be positive")

    campaign_dir = make_run_dir("experiments", f"kernel_{args.dataset}")
    runs = []
    for repeat in range(args.repeats):
        seed = args.seed + repeat
        repeat_dir = os.path.join(campaign_dir, f"repeat_{repeat:02d}_s{seed}")
        os.makedirs(repeat_dir, exist_ok=False)
        runs.append(run_once(args, seed, repeat_dir))

    analysis = {
        metric: paired_comparison(
            [run["quantum"][metric] for run in runs],
            [run["classical"][metric] for run in runs],
            seed=args.seed,
        )
        for metric in ("fisher", "ratio")
    }
    save_json(
        os.path.join(campaign_dir, "campaign_results.json"),
        {
            "schema_version": "1.0",
            "experiment": "04_quantum_kernel",
            "seeds": [run["seed"] for run in runs],
            "runs": runs,
            "paired_statistics": analysis,
        },
    )


if __name__ == "__main__":
    main()
