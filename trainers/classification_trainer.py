import torch
from torch.utils.data import DataLoader
from .trainer import Trainer


# TODO: Implement the classification trainer (issue #4)
class ClassificationTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, train_dataset: DataLoader, valid_dataset: DataLoader):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train(self):
        pass

    def test(self):
        pass