import pytest
import torch
import numpy as np
import sys

from datasets import DatasetFactory


class TestDatasetFactory:

    # DatasetFactory config
    dataset_name = 'svhn'
    imbalance_ratio = 10
    cache_path = 'output'
    validation_split = 0.7
    batch_size = 100
    classes = [0, 1]

    @pytest.fixture(scope="class")
    def dataset_factory(self):
        train_loader, valid_loader, test_loader = DatasetFactory.create(dataset_name=self.dataset_name,
                                                                        imbalance_ratio=self.imbalance_ratio,
                                                                        cache_path=self.cache_path,
                                                                        validation_split=self.validation_split,
                                                                        classes=self.classes,
                                                                        batch_size=self.batch_size)
        yield train_loader, valid_loader, test_loader

    def test_dataset_has_proper_number_of_examples(self, dataset_factory):
        import random
        import torch
        import numpy as np

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        seeded_1_train_loader, seeded_1_valid_loader, seeded_1_test_loader = dataset_factory
        seeded_2_train_loader, seeded_2_valid_loader, seeded_2_test_loader = dataset_factory

        assert len(seeded_1_train_loader.dataset)==len(seeded_2_train_loader.dataset)
        assert len(seeded_1_valid_loader.dataset)==len(seeded_2_valid_loader.dataset)
        assert len(seeded_1_test_loader.dataset)==len(seeded_2_test_loader.dataset)

        s1_dataloader_iter = iter(seeded_1_valid_loader)
        s2_dataloader_iter = iter(seeded_2_valid_loader)
        while True:
            try:
                _, s1_valid_labels = next(s1_dataloader_iter)
                _, s2_valid_labels = next(s2_dataloader_iter)
            except:
                break
        _, s1_last_batch_counts = np.unique(s1_valid_labels, return_counts=True)
        _, s2_last_batch_counts = np.unique(s2_valid_labels, return_counts=True)

        # All diffrent
        assert len(s1_valid_labels)==len(s2_valid_labels)

    def test_dataset_has_proper_shape(self, dataset_factory):
        train_loader, valid_loader, test_loader = dataset_factory
        train_data, train_labels = next(iter(train_loader))
        valid_data, valid_labels = next(iter(valid_loader))
        test_data, test_labels = next(iter(test_loader))

        assert train_data.shape == torch.Size([self.batch_size, 3, 32, 32])
        assert valid_data.shape == torch.Size([self.batch_size, 3, 32, 32])
        assert test_data.shape == torch.Size([self.batch_size, 3, 32, 32])
        assert train_labels.shape == torch.Size([self.batch_size])
        assert valid_labels.shape == torch.Size([self.batch_size])
        assert test_labels.shape == torch.Size([self.batch_size])

    def test_dataset_has_proper_classes(self, dataset_factory):
        train_loader, valid_loader, test_loader = dataset_factory

        # Check that only the right classes are selected
        for loader in (train_loader, valid_loader, test_loader):
            for _, labels in train_loader:
                for label in labels:
                    assert label in self.classes

    def test_dataset_has_proper_imbalance(self, dataset_factory):
        train_loader, valid_loader, _ = dataset_factory

        label_count = {self.classes[0]: 0, self.classes[1]: 1}
        for loader in (train_loader, valid_loader):
            for _, labels in loader:
                for label in labels:
                    label_count[label.item()] += 1

        observed_imbalance_ratio = label_count[self.classes[0]] / label_count[self.classes[1]]
        assert observed_imbalance_ratio == pytest.approx(self.imbalance_ratio, abs=1e-1)

    def test_test_dataset_has_no_imbalance(self, dataset_factory):
        _, _, test_loader = dataset_factory

        label_count = {self.classes[0]: 0, self.classes[1]: 1}
        for _, labels in test_loader:
            for label in labels:
                label_count[label.item()] += 1

        observed_imbalance_ratio = label_count[self.classes[0]] / label_count[self.classes[1]]
        assert observed_imbalance_ratio == pytest.approx(1, abs=1e-1)
