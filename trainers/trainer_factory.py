from typing import List

import torch
from torch.utils.data import DataLoader

from utils import Config
from .classification_trainer import ClassificationTrainer
from .gan_trainer import GANTrainer
from .trainer import Trainer


class TrainerFactory:
    def create(trainer_config: Config.Trainer, model: torch.nn.Module, train_dataset: DataLoader,
               valid_dataset: DataLoader, classes: List[int]) -> Trainer:
        if trainer_config.task == 'classification':
            trainer = ClassificationTrainer(trainer_config=trainer_config,
                                            model=model,
                                            train_dataset=train_dataset,
                                            valid_dataset=valid_dataset,
                                            classes=classes)
        elif trainer_config.task == 'generation':
            trainer = GANTrainer(trainer_config=trainer_config, model=model, dataset=train_dataset, classes=classes)
        return trainer