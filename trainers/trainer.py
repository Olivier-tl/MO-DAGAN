import abc

import torch
from torch.utils.data import DataLoader


# TODO: (Suggestion, not necessarily needed) Implement an abstract trainer class that
#        the classification trainer and gan trainer will inherit from.
class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, model: torch.nn.Module, train_dataset: DataLoader, valid_dataset: DataLoader):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass
