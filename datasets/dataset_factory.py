import os
import typing

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split

from .imbalanced_dataset_sampler import ImbalancedDatasetSampler


class DatasetFactory:
    """Constructs the appropriate dataset. Returns train, valid & test splits.
    """
    def create(dataset_name: str, imbalance_ratio: int, cache_path: str, validation_split: float,
               classes: typing.List[int]) -> typing.Tuple[DataLoader, DataLoader, DataLoader]:
        save_path = os.path.join(cache_path, dataset_name)
        if dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(save_path, train=True, download=True)
            test_dataset = torchvision.datasets.MNIST(save_path, train=False, download=True)
        elif dataset_name == 'fashion-mnist':
            dataset = torchvision.datasets.FashionMNIST(save_path, train=True, download=True)
            test_dataset = torchvision.datasets.FashionMNIST(save_path, train=False, download=True)
        elif dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(save_path, train=True, download=True)
            test_dataset = torchvision.datasets.CIFAR10(save_path, train=False, download=True)
        elif dataset_name == 'svhn':
            dataset = torchvision.datasets.SVHN(save_path, split='train', download=True)
            test_dataset = torchvision.datasets.SVHN(save_path, split='test', download=True)
        else:
            raise ValueError(f'dataset_name "{dataset_name}" not recognized.')

        # Split dataset into train/valid
        valid_length = int(validation_split * len(dataset))
        train_dataset, valid_dataset = random_split(dataset, [len(dataset) - valid_length, valid_length])

        # Create samplers
        train_sampler = ImbalancedDatasetSampler(train_dataset, imbalance_ratio, classes)
        valid_sampler = ImbalancedDatasetSampler(valid_dataset, imbalance_ratio, classes)
        test_sampler = ImbalancedDatasetSampler(test_dataset, imbalance_ratio=1, classes=classes)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, sampler=train_sampler)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler)
        test_loader = DataLoader(test_dataset, sampler=test_sampler)

        return train_loader, valid_loader, test_loader