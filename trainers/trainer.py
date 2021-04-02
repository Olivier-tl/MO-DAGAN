import os

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
    def __init__(self, model: torch.nn.Module, loss: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss = self._get_loss(loss)

    def _get_optimizer(self, opt, model, lr):
        # NOTE: Might need to set more parameters
        if opt == "adam":
            return optim.Adam(model.parameters(), lr)
        elif opt == "sgd":
            return optim.SGD(model.parameters(), lr)
        else:
            logger.warning(f'Optimizer "{opt}" not recognized. Falling back to adam by default')
            return optim.Adam(self.model.parameters(), self.lr)

    def _get_loss(self, opt):
        # NOTE: Might need to set more possible losses/criterions
        if opt == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            logger.warning('Loss "{opt}" not recognized. Falling back to cross_entropy loss by default')
            return nn.crossEntropyLoss()

    def save_model(self, desc: str = ''):
        path = os.path.join(SAVED_MODELS_PATH, self.model.__class__.__name__)
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_model(os.path.join(path, f'{self.model.__class__.__name__}_{desc}'))
