import torch

from .classification import EfficientNetWrapper
from .gans import WGAN
from utils import Config


class ModelFactory:
    def create(model_config: Config.Model) -> torch.nn.Module:
        if model_config.name == 'efficientnet':
            model = EfficientNetWrapper.from_name('efficientnet-b7',
                                                  in_channels=model_config.input_channels,
                                                  image_size=model_config.output_size)
        elif model_config.name == 'wgan':
            model = WGAN(channels=model_config.input_channels, img_shape=model_config.output_size)
        else:
            raise ValueError(f'model name "{model_config.name}" ' f'in config not recognized.')

        if model_config.saved_model != '':
            model.load_model(model_config.saved_model)

        return model