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
    def __init__(self, model: torch.nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

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
            return nn.CrossEntropyLoss()
        else:
            logger.warning('Loss "{loss}" not recognized. Falling back to cross_entropy loss by default')
            return nn.crossEntropyLoss()

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def save_model(self, desc: str = ''):
        path = os.path.join(SAVED_MODELS_PATH, f'{self.model.__class__.__name__}_{self.dataset_name}')
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save_model(os.path.join(path, f'{desc}'))

    def load_model(self, path: str):
        self.model.load_model(path)
