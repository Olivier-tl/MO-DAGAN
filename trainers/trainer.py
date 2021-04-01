import os

import torch
from torch.utils.data import DataLoader

from utils import logging

logger = logging.getLogger()

SAVED_MODELS_PATH = 'output/saved_models'


class Trainer():
    """Parent to the trainers. Implement methods that are common across trainers.
    """
    def __init__(self, lr: float, optimizer: str, loss: str):
        self.model = model
        self.lr = lr
        self.optimizer = self._get_optimizer(optimizer)
        self.loss = self._get_loss(loss)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_optimizer(self, opt):
        # NOTE: Might need to set more parameters
        if opt == "adam":
            return optim.Adam(self.model.parameters(), self.lr)
        elif opt == "sgd":
            return optim.SGD(self.model.parameters(), self.lr)
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

    def save_model(self, epoch):
        if not os.path.exists(SAVED_MODELS_PATH):
            os.mkdir(SAVED_MODELS_PATH)
        path = os.path.join(SAVED_MODELS_PATH, self.model.__class__.__name__, f'_{epoch}')
        self.model.save_model(path)
