import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, random_split


def get_torchvision_loaders(config):
    name = config.dataset_name.lower()

    if name == "cifar10":
        ds_class = torchvision.datasets.CIFAR10
        config.n_channels, config.n_classes = 3, 10
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif name == "fashionmnist":
        ds_class = torchvision.datasets.FashionMNIST
        config.n_channels, config.n_classes = 1, 10
        mean, std = (0.5,), (0.5,)
    elif name == "mnist":
        ds_class = torchvision.datasets.MNIST
        config.n_channels, config.n_classes = 1, 10
        mean, std = (0.5,), (0.5,)
    else:
        raise ValueError(
            f"Dataset '{name}' is not supported by the torchvision loader."
        )

    transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_ds = ds_class(root="./data", train=True, download=True, transform=transform)
    test_ds = ds_class(root="./data", train=False, download=True, transform=transform)

    val_size = max(1, int(0.1 * len(train_ds)))
    train_size = len(train_ds) - val_size
    split_generator = torch.Generator().manual_seed(getattr(config, "seed", 42))
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size], generator=split_generator
    )

    print(
        f"Loaded {name}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    loader_generator = torch.Generator().manual_seed(getattr(config, "seed", 42))
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
