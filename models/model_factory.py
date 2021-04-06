import torch

from .classification import EfficientNetWrapper
from .gans import WGAN
from utils import AttrDict


class ModelFactory:
    def create(model_config: dict) -> torch.nn.Module:
        if model_config['name'] == 'efficientnet':
            model = EfficientNetWrapper.from_name('efficientnet-b7', model_config['in_dim'])
        elif model_config['name'] == 'wgan':
            model = WGAN(AttrDict(model_config['args']), in_dim=model_config['in_dim'])
        else:
            raise ValueError(f'model name "{model_config["name"]}" ' f'in config not recognized.')
        return model