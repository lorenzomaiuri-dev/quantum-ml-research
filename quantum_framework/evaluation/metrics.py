"""
Classification metrics used across experiments 02 and 03.

All functions operate on plain Python lists or NumPy arrays, so they can be
called from any training loop regardless of the framework used.

Public API:

    compute_classification_metrics(labels, preds, probs) → dict
        Accuracy, macro-AUC, macro-F1. The standard evaluation payload
        returned by Trainer.evaluate() in experiments 02 and 03.

    generalization_gap(train_acc, test_acc) → float
        Signed difference train_acc − test_acc. The PRIMARY metric in
        experiment 03: a lower gap for quantum_reg than for bounded_mlp
        (across datasets and seeds) is the evidence for quantum regularisation.

Activation and gradient diagnostics (saturation fraction, gradient L2 norm)
are computed inline inside the trainer using PyTorch hooks. They are not
standalone functions here because they require access to the live model graph.
"""

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(
    labels: Sequence,
    preds: Sequence,
    probs: Sequence,
) -> dict:
    """
    Compute accuracy, AUC, and F1 from a full-pass evaluation.

    Args:
        labels: Ground-truth class indices, shape (N,).
        preds:  Predicted class indices, shape (N,).
        probs:  Predicted class probabilities, shape (N, n_classes).
                Used for AUC; must be softmax outputs.

    Returns:
        dict with keys:
            "accuracy"  — fraction of correct predictions
            "auc"       — macro one-vs-rest ROC AUC; 0.0 if not computable
                          (e.g. only one class present in a small batch)
            "f1"        — macro F1 score; zero_division=0 for unseen classes
    """
    labels = np.asarray(labels)
    preds  = np.asarray(preds)
    probs  = np.asarray(probs)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        # Raised when a split contains only one class (rare but possible with
        # small val sets and high class imbalance).
        auc = 0.0

    return {"accuracy": acc, "auc": auc, "f1": f1}


def generalization_gap(train_acc: float, test_acc: float) -> float:
    """
    Compute the generalization gap.

    The gap is the signed difference between training and test accuracy.
    A positive gap means the model overfits (train > test). A gap near zero
    means the model generalises well.

    This is the PRIMARY metric of experiment 03: if quantum_reg consistently
    shows a lower gap than bounded_mlp at comparable train_acc values, the
    VQC acts as an implicit regulariser.

    Args:
        train_acc: Training set accuracy (float in [0, 1]).
        test_acc:  Test set accuracy (float in [0, 1]).

    Returns:
        train_acc − test_acc  (positive = overfitting, negative = underfitting)
    """
    return train_acc - test_acc
