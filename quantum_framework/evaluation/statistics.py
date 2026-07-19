"""Statistical summaries for paired experimental comparisons."""

from collections.abc import Sequence

import numpy as np
from scipy import stats


def summarize(values: Sequence[float]) -> dict:
    """Return sample size, mean and sample standard deviation."""
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)) if x.size else None,
        "std": float(np.std(x, ddof=1)) if x.size > 1 else None,
    }


def paired_comparison(
    quantum: Sequence[float],
    classical: Sequence[float],
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Compare paired values using quantum minus classical differences."""
    q = np.asarray(quantum, dtype=float)
    c = np.asarray(classical, dtype=float)
    if q.shape != c.shape or q.ndim != 1 or q.size == 0:
        raise ValueError(
            "paired samples must be non-empty one-dimensional arrays of equal length"
        )

    differences = q - c
    result = {
        "direction": "quantum_minus_classical",
        "quantum": summarize(q),
        "classical": summarize(c),
        "differences": summarize(differences),
        "confidence_level": confidence,
        "mean_difference_ci": None,
        "bootstrap_mean_difference_ci": None,
        "paired_t_test": None,
        "wilcoxon": None,
        "cohens_dz": None,
    }
    if q.size < 2:
        return result

    mean_diff = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1))
    if sd_diff > 0:
        sem = stats.sem(differences)
        ci = stats.t.interval(confidence, df=q.size - 1, loc=mean_diff, scale=sem)
        t_result = stats.ttest_rel(q, c)
        result["mean_difference_ci"] = [float(ci[0]), float(ci[1])]
        result["paired_t_test"] = {
            "statistic": float(t_result.statistic),
            "p_value": float(t_result.pvalue),
        }
        result["cohens_dz"] = mean_diff / sd_diff
    else:
        result["mean_difference_ci"] = [mean_diff, mean_diff]

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, q.size, size=(bootstrap_samples, q.size))
    bootstrap_means = differences[bootstrap_indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    result["bootstrap_mean_difference_ci"] = [
        float(np.quantile(bootstrap_means, alpha)),
        float(np.quantile(bootstrap_means, 1.0 - alpha)),
    ]
    try:
        w_result = stats.wilcoxon(q, c)
        result["wilcoxon"] = {
            "statistic": float(w_result.statistic),
            "p_value": float(w_result.pvalue),
        }
    except ValueError:
        # All paired differences are zero.
        pass
    return result
