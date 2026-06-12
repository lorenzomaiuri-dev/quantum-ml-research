"""
Thin wrapper. The canonical MedMNIST loader lives in quantum_framework.data.

load_dataset(config) is preserved for backwards compatibility with the rest
of experiment 02's code. It delegates to load_medmnist and writes the
dataset metadata (n_channels, n_classes) back onto the config object,
matching the original behaviour that model constructors depend on.
"""

from quantum_framework.data import load_medmnist


def load_dataset(config):
    """
    Load MedMNIST and update config.n_channels / config.n_classes in place.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader, val_loader, test_loader, info = load_medmnist(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
    )
    # Write back so model constructors that read config.n_classes still work.
    config.n_channels = info.n_channels
    config.n_classes = info.n_classes
    return train_loader, val_loader, test_loader
