import torch

from .classification import EfficientNetWrapper
from .gans import WGAN

class ModelFactory:
    def create(model_name: str) -> torch.nn.Module:
        if model_name == 'efficientnet':
            model = EfficientNetWrapper('efficientnet-b7')
        elif model_name == 'wgan':
            model = WGAN()
        else:
            raise ValueError(f'model_name "{config["model_name"]}" '
                                f'in config "{config_path}" not recognized.')
        return model