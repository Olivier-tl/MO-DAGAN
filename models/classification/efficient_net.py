import os

import torch
from efficientnet_pytorch import EfficientNet

MODEL_NAME = 'best.pt'


class EfficientNetWrapper(EfficientNet):
    """Light wrapper around EfficientNet to add model saving/loading.
    """
    def __init__(self, model: EfficientNet):
        self.__class__ = type(model.__class__.__name__, (self.__class__, model.__class__), {})
        self.__dict__ = model.__dict__

    def save_model(self, path: str):
        torch.save(self.state_dict(), f'{path}.pt')

    def load_model(self, path: str):
        path = os.path.join(path, MODEL_NAME)
        self.load_state_dict(torch.load(path))

    def from_name(name: str, in_channels: int, num_classes: int) -> EfficientNet:
        return EfficientNetWrapper(EfficientNet.from_name(name, in_channels=in_channels, num_classes=num_classes))
