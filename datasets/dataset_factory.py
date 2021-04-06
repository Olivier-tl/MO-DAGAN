import os
import typing

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from .imbalanced_dataset import ImbalancedDataset
from .synthetic_dataset import SyntheticDataset
from .balanced_dataset import BalancedDataset

CACHE_FOLDER = 'datasets'


class DatasetFactory:
    """Constructs the appropriate dataset. Returns train, valid & test splits.
    """
    def create(dataset_name: str, imbalance_ratio: int, cache_path: str, validation_split: float,
               classes: typing.List[int], batch_size: int,
               oversampling: str) -> typing.Tuple[DataLoader, DataLoader, DataLoader]:
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

        # Create imbalanced dataset
        train_dataset = ImbalancedDataset(dataset, imbalance_ratio, classes)
        # valid_dataset = ImbalancedDataset(valid_dataset, imbalance_ratio, classes)
        test_dataset = ImbalancedDataset(test_dataset, imbalance_ratio=1, classes=classes)

        # Split dataset into train/valid
        valid_length = int(validation_split * len(train_dataset))
        train_dataset, valid_dataset = random_split(train_dataset, [len(train_dataset) - valid_length, valid_length])

        if oversampling == 'oversampling':
            # TODO: Implement oversampling of the minority class
            raise NotImplemented('Oversampling of the minority class not implemented yet.')
        elif oversampling == 'gan':
            synthetic_dataset = SyntheticDataset()
            train_dataset = BalancedDataset(train_dataset, synthetic_dataset)
            valid_dataset = BalancedDataset(valid_dataset, synthetic_dataset)
        elif oversampling == 'none':
            pass  # Do nothing
        else:
            raise ValueError(f'Oversampling option "{oversampling}" not recognized.')

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, valid_loader, test_loader