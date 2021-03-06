import os
from typing import List

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import logging

logger = logging.getLogger()

SAVED_MODELS_PATH = 'output/saved_models'


class Trainer():
    """Parent to the trainers. Implement methods that are common across trainers.
    """
    def __init__(self, model: torch.nn.Module, classes: List[int]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.classes = classes

    def _get_optimizer(self, opt, model, lr, betas):
        # NOTE: Might need to set more parameters
        if opt == "adam":
            return optim.Adam(model.parameters(), lr, betas)
        elif opt == "sgd":
            return optim.SGD(model.parameters(), lr)
        else:
            logger.warning(f'Optimizer "{opt}" not recognized. Falling back to adam by default')
            return optim.Adam(self.model.parameters(), self.lr)

    def _get_loss(self, loss):
        # NOTE: Might need to set more possible losses/criterions
        if loss == "cross_entropy":
            return nn.CrossEntropyLoss(reduction='sum')
        else:
            logger.warning('Loss "{loss}" not recognized. Falling back to cross_entropy loss by default')
            return nn.crossEntropyLoss()

    def save_model(self, desc: str):
        self.model.save_model(desc=desc)

    def load_model(self):
        self.model.load_model()
