"""
Thin wrapper. The canonical MedMNIST loader lives in quantum_framework.data.

load_dataset(config) and make_noisy_loader() are preserved for backwards
compatibility with experiment 03's trainer and main.py.

load_dataset writes n_channels and n_classes back onto the config object
(same behaviour as the original implementation) and supports config.train_subset
for the scarce-data / overfitting regime.
"""

from quantum_framework.data import load_medmnist, make_noisy_loader  # noqa: F401


def load_dataset(config):
    """
    Load MedMNIST and update config.n_channels / config.n_classes in place.

    If config.train_subset > 0, draws a stratified subset of that size
    from the training set (see quantum_framework.data.load_medmnist).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader, val_loader, test_loader, info = load_medmnist(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        train_subset=config.train_subset,
    )
    config.n_channels = info.n_channels
    config.n_classes = info.n_classes
    return train_loader, val_loader, test_loader
