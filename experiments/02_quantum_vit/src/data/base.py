from .medmnist import get_medmnist_loaders
from .torchvision import get_torchvision_loaders

_TORCHVISION_DATASETS = {"cifar10", "fashionmnist", "mnist"}


def get_dataloaders(config):
    """Dispatch to the correct loader based on dataset name."""
    if config.dataset_name.lower() in _TORCHVISION_DATASETS:
        return get_torchvision_loaders(config)
    return get_medmnist_loaders(config)
