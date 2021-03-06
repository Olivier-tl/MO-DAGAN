import torch

from .classification import EfficientNetWrapper
from .gans import WGAN
from utils import Config


class ModelFactory:
    def create(model_config: Config.Model, n_classes: int) -> torch.nn.Module:
        if model_config.name == 'efficientnet':
            model = EfficientNetWrapper.from_name('efficientnet-b7',
                                                  in_channels=model_config.input_channels,
                                                  num_classes=n_classes,
                                                  saving_path=model_config.saved_model)
        elif model_config.name == 'wgan':
            model = WGAN(channels=model_config.input_channels, saving_path=model_config.saved_model)
        else:
            raise ValueError(f'model name "{model_config.name}" in config not recognized.')

        if model_config.load:
            model.load_model()

        return model