# coding: utf-8
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


def get_mnist_dataloader(_mode: str = "train", batch_size: int = 32) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    if _mode == "train":
        _dataset = MNIST("", train=True, download=True, transform=transform)
        _dataset, _ = random_split(
            _dataset, 
            (
                int(len(_dataset) * 0.9), 
                len(_dataset) - int(len(_dataset) * 0.9),
            ),
        )
    elif _mode == "val":
        _dataset = MNIST("", train=True, download=True, transform=transform)
        _, _dataset = random_split(
            _dataset, 
            (
                int(len(_dataset) * 0.9), 
                len(_dataset) - int(len(_dataset) * 0.9),
            ),
        )
    elif _mode == "test":
        _dataset = MNIST("", train=False, download=True, transform=transform)

    return DataLoader(_dataset, batch_size=batch_size)


def get_cifar10_dataloader(_mode: str = "train", batch_size: int = 32) -> DataLoader:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    if _mode == "train":
        _dataset = CIFAR10("CIFAR10", train=True, download=True, transform=transform)
        _dataset, _ = random_split(
            _dataset, 
            (
                int(len(_dataset) * 0.9), 
                len(_dataset) - int(len(_dataset) * 0.9),
            ),
        )
    elif _mode == "val":
        _dataset = CIFAR10("CIFAR10", train=True, download=True, transform=transform)
        _, _dataset = random_split(
            _dataset, 
            (
                int(len(_dataset) * 0.9), 
                len(_dataset) - int(len(_dataset) * 0.9),
            ),
        )
    elif _mode == "test":
        _dataset = CIFAR10("CIFAR10", train=False, download=True, transform=transform)

    return DataLoader(_dataset, batch_size=batch_size)
