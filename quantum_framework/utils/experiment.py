"""
Experiment utilities: reproducibility, run directory management, result saving.

These helpers appear in variants across all four experiments. Centralising them
here ensures consistent behaviour (same timestamp format, same seed coverage)
and makes the thesis code easier to follow.
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for full reproducibility across Python, NumPy, and PyTorch.

    Also disables CUDA non-deterministic algorithms when a GPU is available.

    Args:
        seed: Integer seed value (e.g. 42, 137, 256).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Trade a small performance cost for deterministic convolution kernels.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_run_dir(base: str, tag: str, timestamp_format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Create and return a timestamped directory for one training run.

    The directory is created immediately so that log files can be written
    before training starts.

    Args:
        base:             Parent directory (e.g. "experiments").
        tag:              Human-readable label (e.g. "quantum_reg_pathmnist_s42").
        timestamp_format: strftime format string. Default produces YYYYMMDD_HHMMSS.

    Returns:
        Absolute path to the newly created run directory.

    Example:
        run_dir = make_run_dir("experiments", "quantum_reg_pathmnist_s42")
        # → "experiments/quantum_reg_pathmnist_s42_20260611_143022"
    """
    timestamp = datetime.now().strftime(timestamp_format)
    run_dir = os.path.join(base, f"{tag}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path: str, data: dict) -> None:
    """
    Write a dictionary to a JSON file with 4-space indentation.

    Args:
        path: Full file path (e.g. os.path.join(run_dir, "results.json")).
        data: JSON-serializable dictionary.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
