"""Held-out classifier validation for experiment 04.

This analysis complements the in-sample separation metrics in ``main.py``.
Preprocessing and SVC hyperparameter selection use only the sampled training
data; final metrics are computed on the untouched official MedMNIST test split.
"""

import argparse
import os
from datetime import datetime

import medmnist
import numpy as np
import pennylane as qml
from medmnist import INFO
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from quantum_framework.evaluation import paired_comparison
from quantum_framework.utils import collect_run_metadata, save_json


C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def load_split(dataset_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened images in [0, 1] and integer labels."""
    info = INFO[dataset_name]
    data_class = getattr(medmnist, info["python_class"])
    dataset = data_class(split=split, download=True, size=28)
    images = np.asarray(
        [np.asarray(image).reshape(-1) / 255.0 for image, _ in dataset]
    )
    labels = np.asarray([int(np.asarray(label).squeeze()) for _, label in dataset])
    return images, labels


def sample_training_indices(
    labels: np.ndarray,
    n_class0: int,
    n_class1: int,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic, ordered class-stratified training subset."""
    rng = np.random.default_rng(seed)
    class0 = np.flatnonzero(labels == 0)
    class1 = np.flatnonzero(labels == 1)
    if len(class0) < n_class0 or len(class1) < n_class1:
        raise ValueError("requested samples exceed the available training examples")
    return np.concatenate(
        [
            rng.choice(class0, size=n_class0, replace=False),
            rng.choice(class1, size=n_class1, replace=False),
        ]
    )


def preprocess_train_test(
    train_images: np.ndarray,
    test_images: np.ndarray,
    n_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit every preprocessing transform on train and apply it to test."""
    standardizer = StandardScaler().fit(train_images)
    train_standard = standardizer.transform(train_images)
    test_standard = standardizer.transform(test_images)

    pca = PCA(n_components=n_qubits, svd_solver="full").fit(train_standard)
    train_pca = pca.transform(train_standard)
    test_pca = pca.transform(test_standard)

    angle_scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(train_pca)
    train_angles = angle_scaler.transform(train_pca)
    test_angles = angle_scaler.transform(test_pca)
    return (
        train_angles,
        test_angles,
        train_pca,
        test_pca,
        float(pca.explained_variance_ratio_.sum()),
    )


def iqp_states(features: np.ndarray, n_qubits: int, repeats: int) -> np.ndarray:
    """Build exact statevectors for the independently implemented IQP map."""
    device = qml.device("default.qubit", wires=n_qubits)
    pattern = [[wire, wire + 1] for wire in range(n_qubits - 1)]

    @qml.qnode(device)
    def circuit(x):
        qml.IQPEmbedding(
            x,
            wires=range(n_qubits),
            n_repeats=repeats,
            pattern=pattern,
        )
        return qml.state()

    return np.asarray([circuit(row) for row in features])


def fidelity_kernel(left_states: np.ndarray, right_states: np.ndarray) -> np.ndarray:
    """Return |<left|right>|^2 for two batches of statevectors."""
    return np.abs(left_states.conj() @ right_states.T) ** 2


def median_rbf_gamma(train_features: np.ndarray) -> float:
    """Select an RBF bandwidth without labels or test data."""
    distances = np.sum(
        (train_features[:, np.newaxis, :] - train_features[np.newaxis, :, :])
        ** 2,
        axis=-1,
    )
    upper = distances[np.triu_indices_from(distances, k=1)]
    median_squared_distance = float(np.median(upper))
    if median_squared_distance <= 0:
        raise ValueError("cannot select an RBF bandwidth from zero distances")
    return 1.0 / (2.0 * median_squared_distance)


def select_svc_c(kernel: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """Choose C using five-fold training-only ROC-AUC cross-validation."""
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {}
    for c_value in C_GRID:
        fold_scores = []
        for train_idx, val_idx in splitter.split(kernel, labels):
            model = SVC(kernel="precomputed", C=c_value, class_weight="balanced")
            model.fit(kernel[np.ix_(train_idx, train_idx)], labels[train_idx])
            decision = model.decision_function(kernel[np.ix_(val_idx, train_idx)])
            fold_scores.append(roc_auc_score(labels[val_idx], decision))
        scores[str(c_value)] = [float(score) for score in fold_scores]

    # max() is stable over C_GRID order, so ties prefer the smaller C.
    selected = max(C_GRID, key=lambda value: np.mean(scores[str(value)]))
    return {
        "selected_c": selected,
        "mean_cv_auc": float(np.mean(scores[str(selected)])),
        "fold_auc_by_c": scores,
    }


def evaluate_kernel_classifier(
    train_kernel: np.ndarray,
    test_train_kernel: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> dict:
    """Tune on training folds, refit on all training data, evaluate on test."""
    tuning = select_svc_c(train_kernel, train_labels, seed)
    model = SVC(
        kernel="precomputed",
        C=tuning["selected_c"],
        class_weight="balanced",
    )
    model.fit(train_kernel, train_labels)
    predictions = model.predict(test_train_kernel)
    decision = model.decision_function(test_train_kernel)
    return {
        **tuning,
        "test_accuracy": float(accuracy_score(test_labels, predictions)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, predictions)
        ),
        "test_auc": float(roc_auc_score(test_labels, decision)),
        "test_macro_f1": float(
            f1_score(test_labels, predictions, average="macro", zero_division=0)
        ),
        "test_predictions": [int(value) for value in predictions],
        "test_decision_scores": [float(value) for value in decision],
    }


def run_once(args, train_images, train_labels, test_images, test_labels, seed):
    indices = sample_training_indices(
        train_labels, args.n_class0, args.n_class1, seed
    )
    sampled_images = train_images[indices]
    sampled_labels = train_labels[indices]
    train_angles, test_angles, train_pca, test_pca, explained = (
        preprocess_train_test(sampled_images, test_images, args.n_qubits)
    )

    train_states = iqp_states(train_angles, args.n_qubits, args.iqp_repeats)
    test_states = iqp_states(test_angles, args.n_qubits, args.iqp_repeats)
    quantum_train = fidelity_kernel(train_states, train_states)
    quantum_test = fidelity_kernel(test_states, train_states)

    cosine_train = cosine_similarity(train_angles)
    cosine_test = cosine_similarity(test_angles, train_angles)
    rbf_gamma = median_rbf_gamma(train_pca)
    rbf_train = rbf_kernel(train_pca, gamma=rbf_gamma)
    rbf_test = rbf_kernel(test_pca, train_pca, gamma=rbf_gamma)

    kernels = {
        "iqp": (quantum_train, quantum_test),
        "cosine_angle": (cosine_train, cosine_test),
        "rbf_median_pca": (rbf_train, rbf_test),
    }
    metrics = {
        name: evaluate_kernel_classifier(
            train_kernel,
            test_kernel,
            sampled_labels,
            test_labels,
            seed,
        )
        for name, (train_kernel, test_kernel) in kernels.items()
    }
    return {
        "seed": seed,
        "training_indices": indices.tolist(),
        "training_class_counts": {
            "0": int(np.sum(sampled_labels == 0)),
            "1": int(np.sum(sampled_labels == 1)),
        },
        "test_class_counts": {
            "0": int(np.sum(test_labels == 0)),
            "1": int(np.sum(test_labels == 1)),
        },
        "pca_solver": "full",
        "pca_explained_variance": explained,
        "rbf_gamma": rbf_gamma,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out kernel SVC validation")
    parser.add_argument("--dataset", default="breastmnist")
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-class0", type=int, default=50)
    parser.add_argument("--n-class1", type=int, default=50)
    parser.add_argument("--iqp-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if min(
        args.n_qubits,
        args.n_class0,
        args.n_class1,
        args.iqp_repeats,
        args.repeats,
    ) <= 0:
        parser.error("dimensions, sample counts and repeats must be positive")

    train_images, train_labels = load_split(args.dataset, "train")
    test_images, test_labels = load_split(args.dataset, "test")
    runs = []
    for offset in range(args.repeats):
        seed = args.seed + offset
        print(f"Validation repeat {offset + 1}/{args.repeats}, seed={seed}")
        runs.append(
            run_once(
                args,
                train_images,
                train_labels,
                test_images,
                test_labels,
                seed,
            )
        )

    metric_names = (
        "test_accuracy",
        "test_balanced_accuracy",
        "test_auc",
        "test_macro_f1",
    )
    comparisons = {}
    for baseline in ("cosine_angle", "rbf_median_pca"):
        comparisons[f"iqp_vs_{baseline}"] = {
            metric: paired_comparison(
                [run["metrics"]["iqp"][metric] for run in runs],
                [run["metrics"][baseline][metric] for run in runs],
                seed=args.seed,
            )
            for metric in metric_names
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        "experiments",
        f"classifier_validation_{args.dataset}_{args.n_class0}_{args.n_class1}_{timestamp}",
    )
    os.makedirs(output_dir, exist_ok=False)
    payload = {
        "schema_version": "1.1",
        "experiment": "04_quantum_kernel_classifier_validation",
        "dataset": args.dataset,
        "feature_map": "iqp",
        "iqp_repeats": args.iqp_repeats,
        "n_qubits": args.n_qubits,
        "n_class0": args.n_class0,
        "n_class1": args.n_class1,
        "test_split": "official_test",
        "label_mapping": INFO[args.dataset]["label"],
        "test_labels": [int(value) for value in test_labels],
        "svc_c_grid": list(C_GRID),
        "seeds": [run["seed"] for run in runs],
        "runs": runs,
        "paired_statistics": comparisons,
    }
    save_json(os.path.join(output_dir, "validation_results.json"), payload)
    save_json(
        os.path.join(output_dir, "run_manifest.json"),
        collect_run_metadata(
            "04_quantum_kernel_classifier_validation",
            args.seed,
            "iqp_vs_cosine_vs_rbf",
        ),
    )
    print(f"Saved validation campaign to {output_dir}")


if __name__ == "__main__":
    main()
