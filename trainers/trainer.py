import abc

import torch

# TODO: (Suggestion, not necessarily needed) Implement an abstract trainer class that 
#        the classification trainer and gan trainer will inherit from.  
class Trainer(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.DataLoader):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def test(self):
        pass

