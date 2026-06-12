import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
        raise ValueError(f"Dataset '{name}' is not supported by the torchvision loader.")

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ds_class(root="./data", train=True, download=True, transform=transform)
    test_ds = ds_class(root="./data", train=False, download=True, transform=transform)

    print(f"Loaded {name}: train={len(train_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    # test_loader doubles as val_loader for these standard datasets
    return train_loader, test_loader, test_loader
