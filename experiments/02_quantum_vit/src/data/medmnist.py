"""
Thin wrapper. The canonical MedMNIST loader lives in quantum_framework.data.

get_medmnist_loaders(config) is preserved for backwards compatibility with
src/data/base.py which dispatches here for MedMNIST datasets.
"""

from quantum_framework.data import load_medmnist


def get_medmnist_loaders(config):
    """Delegates to load_medmnist and writes metadata back onto config."""
    train_loader, val_loader, test_loader, info = load_medmnist(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        train_subset=config.train_subset,
        seed=config.seed,
    )
    config.n_channels = info.n_channels
    config.n_classes = info.n_classes
    return train_loader, val_loader, test_loader
