import os
import typing

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from utils import Config
from .imbalanced_dataset import ImbalancedDataset
from .synthetic_dataset import SyntheticDataset
from .balanced_dataset import BalancedDataset

CACHE_FOLDER = 'output/datasets'


class DatasetFactory:
    """Constructs the appropriate dataset. Returns train, valid & test splits.
    """
    def create(dataset_config: Config.Dataset) -> typing.Tuple[DataLoader, DataLoader, DataLoader]:
        save_path = os.path.join(CACHE_FOLDER, dataset_config.name)

        # FIXME : Normalize using the actual mean and std of the dataset (issue #15)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

        if dataset_config.name == 'mnist':
            dataset = torchvision.datasets.MNIST(save_path, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(save_path, train=False, download=True, transform=transform)
        elif dataset_config.name == 'fashion-mnist':
            dataset = torchvision.datasets.FashionMNIST(save_path, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(save_path, train=False, download=True, transform=transform)
        elif dataset_config.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(save_path, train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.CIFAR10(save_path, train=False, download=True, transform=transform)
        elif dataset_config.name == 'svhn':
            dataset = torchvision.datasets.SVHN(save_path, split='train', download=True, transform=transform)
            test_dataset = torchvision.datasets.SVHN(save_path, split='test', download=True, transform=transform)
        else:
            raise ValueError(f'dataset_name "{dataset_config.name}" not recognized.')

        # Create imbalanced dataset
        train_dataset = ImbalancedDataset(dataset, dataset_config.imbalance_ratio, dataset_config.classes)
        test_dataset = ImbalancedDataset(test_dataset, imbalance_ratio=1, classes=dataset_config.classes)

        # Split dataset into train/valid
        valid_length = int(dataset_config.validation_split * len(train_dataset))
        train_dataset, valid_dataset = random_split(train_dataset, [len(train_dataset) - valid_length, valid_length])

        if dataset_config.oversampling == 'oversampling':
            # TODO: Implement oversampling of the minority class
            raise NotImplemented('Oversampling of the minority class not implemented yet.')
        elif dataset_config.oversampling == 'gan':
            synthetic_dataset = SyntheticDataset(dataset_config.gan_model)
            train_dataset = BalancedDataset(train_dataset, synthetic_dataset)
            valid_dataset = BalancedDataset(valid_dataset, synthetic_dataset)
        elif dataset_config.oversampling == 'none':
            pass  # Do nothing
        else:
            raise ValueError(f'Oversampling option "{oversampling}" not recognized.')

        train_loader = DataLoader(train_dataset, batch_size=dataset_config.batch_size)
        valid_loader = DataLoader(valid_dataset, batch_size=dataset_config.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=dataset_config.batch_size)

        return train_loader, valid_loader, test_loader