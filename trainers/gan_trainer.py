import torch

from .trainer import Trainer

# TODO: Implement the gan trainer (issue #7)
class GANTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.DataLoader): 
        self.model = model
        self.dataset = dataset

    def train(self):
        pass

    def test(self):
        pass