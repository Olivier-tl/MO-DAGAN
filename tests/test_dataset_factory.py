import pytest
import torch

from datasets import DatasetFactory


def test_dataset_factory():
    train_loader, valid_loader, test_loader = DatasetFactory.create(dataset_name='svhn',
                                                                    imbalance_ratio=10,
                                                                    cache_path='output',
                                                                    validation_split=0.7,
                                                                    classes=[0, 1],
                                                                    batch_size=100)
    # Load one batch
    data, labels = next(iter(train_loader))

    assert data.shape == torch.Size([100, 3, 32, 32])
    assert labels.shape == torch.Size([100])
