"""
MedMNIST data loading utilities shared across experiments 02, 03, and 04.

Public API:

    load_medmnist(dataset_name, batch_size, ...)
        → train_loader, val_loader, test_loader, DatasetInfo

    make_noisy_loader(test_loader, noise_type, intensity, batch_size)
        → DataLoader with noise-corrupted images (used in experiment 03)

Design notes:

- The function signature uses explicit parameters (not a config object) so
  that any experiment can call it regardless of its config class layout.
  Experiments that have a config object pass config.dataset_name, etc.

- Normalisation: pixel values are mapped to approximately [-1, 1] via
  Normalize(mean=0.5, std=0.5) per channel. This is standard for MedMNIST
  and keeps values in a range compatible with tanh-based angle embedding.

- train_subset: when > 0, a stratified subset of the training set is drawn
  using sklearn. Stratification ensures proportional class representation even
  in imbalanced datasets (e.g. PathMNIST). This was introduced in experiment 03
  to study the scarce-data / overfitting regime.

- NoisyDataset: noise is applied at test time only, after training. It is used
  in experiment 03 to evaluate noise robustness as a proxy for generalisation.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

import medmnist
from medmnist import INFO
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------

@dataclass
class DatasetInfo:
    """Metadata about a MedMNIST dataset, populated after loading."""
    name: str
    n_classes: int
    n_channels: int
    n_train: int
    n_val: int
    n_test: int


# ---------------------------------------------------------------------------
# Dataset wrappers
# ---------------------------------------------------------------------------

class MedMNISTWrapper(Dataset):
    """
    Thin wrapper around a MedMNIST split.

    MedMNIST returns labels shaped (N, 1). CrossEntropyLoss expects (N,).
    This wrapper squeezes and casts the label to torch.long.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        return image, torch.tensor(label.squeeze(), dtype=torch.long)


class NoisyDataset(Dataset):
    """
    Wraps a dataset and applies pixel-level noise corruption to images.

    Used for robustness evaluation after training (experiment 03). Noise is
    applied at access time, not pre-computed, so memory cost is constant.

    Supported noise types:
        "gaussian"    — additive Gaussian noise scaled by intensity
        "salt_pepper" — random pixels set to ±1 (min/max after normalisation)
        "blur"        — box-blur via average pooling; kernel size scales with intensity

    Args:
        base_dataset: The clean dataset to wrap.
        noise_type:   One of "gaussian", "salt_pepper", "blur".
        intensity:    Float in [0, 1] controlling noise severity.
    """

    def __init__(self, base_dataset, noise_type: str = "gaussian", intensity: float = 0.1):
        self.base = base_dataset
        self.noise_type = noise_type
        self.intensity = intensity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]

        if self.noise_type == "gaussian":
            noise = torch.randn_like(image) * self.intensity
            image = image + noise

        elif self.noise_type == "salt_pepper":
            mask = torch.rand_like(image)
            image = image.clone()
            image[mask < self.intensity / 2] = -1.0   # salt (normalised min)
            image[mask > 1 - self.intensity / 2] = 1.0  # pepper (normalised max)

        elif self.noise_type == "blur":
            # Box blur via average pooling + upsample to preserve spatial dimensions.
            k = max(3, int(self.intensity * 7))
            if k % 2 == 0:
                k += 1
            padding = k // 2
            image = F.avg_pool2d(image.unsqueeze(0), k, stride=1, padding=padding).squeeze(0)

        return image, label


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_medmnist(
    dataset_name: str,
    batch_size: int,
    train_subset: int = 0,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetInfo]:
    """
    Load a MedMNIST dataset and return train/val/test DataLoaders.

    Args:
        dataset_name:  MedMNIST key, e.g. "pathmnist", "bloodmnist", "dermamnist".
        batch_size:    Number of samples per batch (same for all splits).
        train_subset:  If > 0, draw a stratified subset of this many training
                       samples. Useful for studying the scarce-data regime (exp 03).
                       0 means the full training set.
        seed:          Random seed for the stratified split (only used when
                       train_subset > 0).
        num_workers:   DataLoader worker processes. 0 = load in the main process
                       (safest for PennyLane which uses its own thread pool).

    Returns:
        train_loader, val_loader, test_loader, info

        info is a DatasetInfo with n_classes, n_channels, and split sizes
        that can be used to configure the model (e.g. info.n_classes → head dim).
    """
    meta = INFO[dataset_name]
    n_channels = meta["n_channels"]
    n_classes = len(meta["label"])

    DataClass = getattr(medmnist, meta["python_class"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels),
    ])

    train_ds = MedMNISTWrapper(DataClass(split="train", transform=transform, download=True, size=28))
    val_ds   = MedMNISTWrapper(DataClass(split="val",   transform=transform, download=True, size=28))
    test_ds  = MedMNISTWrapper(DataClass(split="test",  transform=transform, download=True, size=28))

    full_train_size = len(train_ds)

    # Stratified subsetting for the scarce-data / overfitting regime (experiment 03).
    if 0 < train_subset < full_train_size:
        all_labels = np.array([train_ds[i][1].item() for i in range(full_train_size)])
        indices = np.arange(full_train_size)
        subset_indices, _ = train_test_split(
            indices,
            train_size=train_subset,
            stratify=all_labels,
            random_state=seed,
        )
        # Log class distribution for reproducibility audit.
        unique, counts = np.unique(all_labels[subset_indices], return_counts=True)
        dist_str = ", ".join(f"c{u}:{c}" for u, c in zip(unique, counts))
        print(f"  Stratified subset: {train_subset}/{full_train_size} [{dist_str}]")
        train_ds = Subset(train_ds, subset_indices)

    info = DatasetInfo(
        name=dataset_name,
        n_classes=n_classes,
        n_channels=n_channels,
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_test=len(test_ds),
    )

    print(f"  Dataset: {dataset_name}")
    print(f"  Classes: {n_classes} | Channels: {n_channels}")
    print(f"  Train: {info.n_train} | Val: {info.n_val} | Test: {info.n_test}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, info


def make_noisy_loader(
    test_loader: DataLoader,
    noise_type: str,
    intensity: float,
    batch_size: int,
) -> DataLoader:
    """
    Wrap an existing test DataLoader with noise corruption.

    Args:
        test_loader: The clean test loader whose dataset will be wrapped.
        noise_type:  "gaussian" | "salt_pepper" | "blur"
        intensity:   Float in [0, 1] controlling noise severity.
        batch_size:  Batch size for the returned loader.

    Returns:
        A new DataLoader yielding noise-corrupted (image, label) pairs.
    """
    noisy_ds = NoisyDataset(test_loader.dataset, noise_type=noise_type, intensity=intensity)
    return DataLoader(noisy_ds, batch_size=batch_size, shuffle=False, num_workers=0)
