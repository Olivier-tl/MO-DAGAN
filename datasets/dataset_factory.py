import os
import typing

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from .imbalanced_dataset_sampler import ImbalancedDatasetSampler

CACHE_FOLDER = 'datasets'


class DatasetFactory:
    """Constructs the appropriate dataset. Returns train, valid & test splits.
    """
    def create(dataset_name: str, imbalance_ratio: int, force_balance: bool, cache_path: str, validation_split: float,
               classes: typing.List[int], batch_size: int) -> typing.Tuple[DataLoader, DataLoader, DataLoader]:
        save_path = os.path.join(cache_path, CACHE_FOLDER, dataset_name)

        # FIXME : Normalize using the actual mean and std of the dataset (issue #15)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

        if dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST(save_path, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(save_path, train=False, download=True, transform=transform)
        elif dataset_name == 'fashion-mnist':
            dataset = torchvision.datasets.FashionMNIST(save_path, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(save_path, train=False, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(save_path, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(save_path, train=False, download=True, transform=transform)
        elif dataset_name == 'svhn':
            dataset = torchvision.datasets.SVHN(save_path, split='train', download=True, transform=transform)
            test_dataset = torchvision.datasets.SVHN(save_path, split='test', download=True, transform=transform)
        else:
            raise ValueError(f'dataset_name "{dataset_name}" not recognized.')

        # Split dataset into train/valid
        valid_length = int(validation_split * len(dataset))
        train_dataset, valid_dataset = random_split(dataset, [len(dataset) - valid_length, valid_length])

        # Create samplers
        if force_balance:
            train_sampler = get_balanced_sampler(train_dataset.dataset.labels[train_dataset.indices])
            valid_sampler = get_balanced_sampler(valid_dataset.dataset.labels[valid_dataset.indices])
            test_sampler = get_balanced_sampler(test_dataset.labels)
        else:
            train_sampler = ImbalancedDatasetSampler(train_dataset, imbalance_ratio, classes)
            valid_sampler = ImbalancedDatasetSampler(valid_dataset, imbalance_ratio, classes)
            test_sampler = ImbalancedDatasetSampler(test_dataset, imbalance_ratio=1, classes=classes)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        return train_loader, valid_loader, test_loader

def get_balanced_sampler(labels):
    # Create balanced sampler
    class_sample_count = np.array(
        [len(np.where(labels == l)[0]) for l in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[l] for l in labels])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    return WeightedRandomSampler(samples_weight, len(samples_weight))