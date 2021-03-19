import torch

from .classification import EfficientNet
from .gans import WGAN
from utils import AttrDict

class ModelFactory:
    def create(model_config: dict) -> torch.nn.Module:
        model_config['name']
        if model_config['name'] == 'efficientnet':
            model = EfficientNet()
        elif model_config['name'] == 'wgan':
            model = WGAN(AttrDict(model_config['args']))
        else:
            raise ValueError(f'model name "{model_config["name"]}" '
                                f'in config not recognized.')
        return model