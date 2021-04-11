import os
import typing

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from utils import Config
from .imbalanced_dataset import ImbalancedDataset
from .synthetic_dataset import SyntheticDataset
from .balanced_dataset import BalancedDataset

CACHE_FOLDER = 'output/datasets'
IMG_RES = 32


class DatasetFactory:
    """Constructs the appropriate dataset. Returns train, valid & test splits.
    """
    def create(dataset_config: Config.Dataset) -> typing.Tuple[DataLoader, DataLoader, DataLoader]:
        save_path = os.path.join(CACHE_FOLDER, dataset_config.name)

        # FIXME : Normalize using the actual mean and std of the dataset (issue #15)
        transform = transforms.Compose(
            [transforms.Resize(IMG_RES),
             transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))])

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

        train_sampler, valid_sampler, test_sampler = None, None, None
        if dataset_config.oversampling == 'oversampling':
            train_sampler = get_balanced_sampler(train_dataset.dataset.labels[train_dataset.indices])
            valid_sampler = get_balanced_sampler(valid_dataset.dataset.labels[valid_dataset.indices])
            test_sampler = get_balanced_sampler(test_dataset.labels)
        elif dataset_config.oversampling == 'gan':
            synthetic_dataset = SyntheticDataset(dataset_config.gan_model)
            train_dataset = BalancedDataset(train_dataset, synthetic_dataset)
            valid_dataset = BalancedDataset(valid_dataset, synthetic_dataset)
        elif dataset_config.oversampling == 'none':
            pass  # Do nothing
        else:
            raise ValueError(f'Oversampling option "{dataset_config.oversampling}" not recognized.')

        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=dataset_config.batch_size)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=dataset_config.batch_size)
        test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=dataset_config.batch_size)

        return train_loader, valid_loader, test_loader


def get_balanced_sampler(labels):
    # Create balanced sampler
    class_sample_count = np.array([len(np.where(labels == l)[0]) for l in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[l] for l in labels])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    return WeightedRandomSampler(samples_weight, len(samples_weight))