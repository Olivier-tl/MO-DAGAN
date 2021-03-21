import torch
from torch.utils.data import DataLoader

from .classification_trainer import ClassificationTrainer
from .gan_trainer import GANTrainer
from .trainer import Trainer


class TrainerFactory:
    def create(task: str, model: torch.nn.Module, train_dataset: DataLoader, valid_dataset: DataLoader) -> Trainer:
        if task == 'classification':
            trainer = ClassificationTrainer(model=model, train_dataset=train_dataset, valid_dataset=valid_dataset)
        elif task == 'generation':
            trainer = GANTrainer(model=model, dataset=train_dataset)
        return trainer