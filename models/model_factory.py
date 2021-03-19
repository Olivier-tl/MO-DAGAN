import torch

from efficientnet_pytorch import EfficientNet
from .gans import WGAN

class ModelFactory:
    def create(model_name: str) -> torch.nn.Module:
        if model_name == 'efficientnet':
            model = EfficientNet.from_name('efficientnet-b8')
        elif model_name == 'wgan':
            model = WGAN()
        else:
            raise ValueError(f'model_name "{config["model_name"]}" '
                                f'in config "{config_path}" not recognized.')
        return model