import pytest
import torch

from datasets import DatasetFactory
from utils import Config

gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


class TestDatasetFactory:

    # DatasetFactory config
    config = Config.Dataset()
    config.name = 'svhn'
    config.imbalance_ratio = 10
    config.validation_split = 0.3
    config.batch_size = 100
    config.classes = [0, 1]
    gan_model = Config.Dataset.Model()
    gan_model.name = 'wgan'
    gan_model.saved_model = ''
    gan_model.output_size = 32
    gan_model.input_channels = 3
    config.gan_model = gan_model

    @pytest.fixture(scope="class")
    def dataset_factory(self):
        self.config.oversampling = 'none'
        train_loader, valid_loader, test_loader = DatasetFactory.create(dataset_config=self.config)
        yield train_loader, valid_loader, test_loader

    @pytest.fixture(scope="class")
    def dataset_factory_balanced(self):
        self.config.oversampling = 'gan'
        train_loader, valid_loader, test_loader = DatasetFactory.create(dataset_config=self.config)
        yield train_loader, valid_loader, test_loader

    def test_dataset_has_proper_shape(self, dataset_factory):
        train_loader, valid_loader, test_loader = dataset_factory
        train_data, train_labels = next(iter(train_loader))
        valid_data, valid_labels = next(iter(valid_loader))
        test_data, test_labels = next(iter(test_loader))

        assert train_data.shape == torch.Size([self.config.batch_size, 3, 32, 32])
        assert valid_data.shape == torch.Size([self.config.batch_size, 3, 32, 32])
        assert test_data.shape == torch.Size([self.config.batch_size, 3, 32, 32])
        assert train_labels.shape == torch.Size([self.config.batch_size])
        assert valid_labels.shape == torch.Size([self.config.batch_size])
        assert test_labels.shape == torch.Size([self.config.batch_size])

    def test_dataset_has_proper_classes(self, dataset_factory):
        train_loader, valid_loader, test_loader = dataset_factory

        # Check that only the right classes are selected
        for loader in (train_loader, valid_loader, test_loader):
            for _, labels in train_loader:
                for label in labels:
                    assert label in self.config.classes

    def test_dataset_has_proper_imbalance(self, dataset_factory):
        train_loader, valid_loader, _ = dataset_factory

        label_count = {self.config.classes[0]: 0, self.config.classes[1]: 1}
        for loader in (train_loader, valid_loader):
            for _, labels in loader:
                for label in labels:
                    label_count[label.item()] += 1

        observed_imbalance_ratio = label_count[self.config.classes[0]] / label_count[self.config.classes[1]]
        assert observed_imbalance_ratio == pytest.approx(self.config.imbalance_ratio, abs=1e-1)

    def test_test_dataset_has_no_imbalance(self, dataset_factory):
        _, _, test_loader = dataset_factory

        label_count = {self.config.classes[0]: 0, self.config.classes[1]: 1}
        for _, labels in test_loader:
            for label in labels:
                label_count[label.item()] += 1

        observed_imbalance_ratio = label_count[self.config.classes[0]] / label_count[self.config.classes[1]]
        assert observed_imbalance_ratio == pytest.approx(1, abs=1e-1)

    @gpu_only
    def test_oversampling_gan_is_balanced(self, dataset_factory_balanced):
        train_loader, valid_loader, _ = dataset_factory_balanced

        for loader in (train_loader, valid_loader):
            label_count = {self.config.classes[0]: 0, self.config.classes[1]: 1}
            for _, labels in loader:
                for label in labels:
                    label_count[label.item()] += 1

            observed_imbalance_ratio = label_count[self.config.classes[0]] / label_count[self.config.classes[1]]
            assert observed_imbalance_ratio == pytest.approx(1, abs=1e-1)
