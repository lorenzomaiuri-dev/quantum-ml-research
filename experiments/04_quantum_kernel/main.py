import time

import medmnist
import numpy as np
import pennylane as qml
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_samples(dataset_name, n_class0, n_class1, n_qubits):
    """Load and PCA-compress samples from two classes."""
    info = INFO[dataset_name]
    DataClass = getattr(medmnist, info["python_class"])
    ds = DataClass(split="train", download=True, size=28)

    # Separate by class
    images_0, images_1 = [], []
    for img, label in ds:
        img_flat = np.array(img).flatten() / 255.0
        lab = int(label.squeeze())
        if lab == 0 and len(images_0) < n_class0:
            images_0.append(img_flat)
        elif lab == 1 and len(images_1) < n_class1:
            images_1.append(img_flat)
        if len(images_0) >= n_class0 and len(images_1) >= n_class1:
            break

    images_0 = np.array(images_0)
    images_1 = np.array(images_1)

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

    x0 = all_scaled[:len(images_0)]
    x1 = all_scaled[len(images_0):]

    return x0, x1


def classical_similarity(x0, x1):
    """Compute dot-product similarity matrix."""
    all_x = np.vstack([x0, x1])
    # Normalize for cosine similarity
    norms = np.linalg.norm(all_x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    all_normed = all_x / norms
    sim = all_normed @ all_normed.T
    return sim


def quantum_kernel_matrix(x0, x1, n_qubits):
    """
    Compute quantum kernel matrix using IQP-style embedding.

    K(x, y) = |⟨0|U†(y)U(x)|0⟩|²

    where U(x) = H⊗n · diag(exp(i·x)) · CNOT_cascade · H⊗n · diag(exp(i·x))

    This is a standard kernel from the QML literature (Havlicek et al., Nature 2019).
    No trainable parameters — purely a function of the data.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

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
                print(f"  {computed}/{total_pairs} pairs ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    return K


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
    block_11 = sim_matrix[n0:n0 + n1, n0:n0 + n1]
    block_01 = sim_matrix[:n0, n0:n0 + n1]

    # Remove diagonal from intra-class
    intra_0_vals = block_00[np.triu_indices_from(block_00, k=1)]
    intra_1_vals = block_11[np.triu_indices_from(block_11, k=1)]
    inter_vals = block_01.flatten()

    intra_0 = np.mean(intra_0_vals)
    intra_1 = np.mean(intra_1_vals)
    inter = np.mean(inter_vals)

    intra_mean = (intra_0 + intra_1) / 2
    ratio = intra_mean / inter if inter > 1e-10 else float('inf')

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
        "intra_0": round(float(intra_0), 4),
        "intra_1": round(float(intra_1), 4),
        "inter": round(float(inter), 4),
        "ratio": round(float(ratio), 4),
        "fisher": round(float(fisher), 4),
    }


def main():
    N_QUBITS = 8
    N_CLASS0 = 80
    N_CLASS1 = 20

    print("=" * 50)
    print("  QUANTUM KERNEL PoC")
    print("  Does quantum similarity separate classes better")
    print("  than classical dot product?")
    print("=" * 50)

    # Load data
    print("\n--- Loading data ---")
    x0, x1 = load_samples("breastmnist", N_CLASS0, N_CLASS1, N_QUBITS)

    # Classical similarity
    print("\n--- Classical Kernel (cosine similarity) ---")
    classical_sim = classical_similarity(x0, x1)
    classical_metrics = analyze_separation(classical_sim, len(x0), len(x1), "CLASSICAL")

    # Quantum kernel
    print("\n--- Quantum Kernel (IQP embedding) ---")
    quantum_sim = quantum_kernel_matrix(x0, x1, N_QUBITS)
    quantum_metrics = analyze_separation(quantum_sim, len(x0), len(x1), "QUANTUM")

    # Verdict
    print(f"\n{'=' * 50}")
    print(f"  VERDICT")
    print(f"{'=' * 50}")

    q_fisher = quantum_metrics["fisher"]
    c_fisher = classical_metrics["fisher"]
    q_ratio = quantum_metrics["ratio"]
    c_ratio = classical_metrics["ratio"]

    print(f"  Fisher discriminant: Classical={c_fisher:.4f} | Quantum={q_fisher:.4f}")
    print(f"  Separation ratio:   Classical={c_ratio:.4f} | Quantum={q_ratio:.4f}")

    if q_fisher > c_fisher * 1.1:
        print(f"\n  ✓ QUANTUM WINS (Fisher +{(q_fisher/c_fisher - 1)*100:.1f}%)")
        print(f"  → Proceed to full Quantum Kernel Attention ViT")
    elif c_fisher > q_fisher * 1.1:
        print(f"\n  ✗ CLASSICAL WINS (Fisher +{(c_fisher/q_fisher - 1)*100:.1f}%)")
        print(f"  → Quantum kernel does not add value on this task")
    else:
        print(f"\n  ≈ NO SIGNIFICANT DIFFERENCE")
        print(f"  → Try different embedding or dataset")

    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()