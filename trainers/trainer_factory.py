import torch

from .classification_trainer import ClassificationTrainer
from .gan_trainer import GANTrainer
from .trainer import Trainer

class TrainerFactory:
    def create(task: str, model: torch.nn.Module, dataset: torch.utils.data.DataLoader) -> Trainer:
        if task == 'classification':
            trainer = ClassificationTrainer(model=model, dataset=dataset)
        elif task == 'generation':
            trainer = GANTrainer(model=model, dataset=dataset)
        return trainer