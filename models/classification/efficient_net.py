import os

import torch
from efficientnet_pytorch import EfficientNet

MODEL_NAME = 'best.pt'


class EfficientNetWrapper(EfficientNet):
    """Light wrapper around EfficientNet to add model saving/loading.
    """
    def __init__(self, model: EfficientNet, saving_path: str):
        self.__class__ = type(model.__class__.__name__, (self.__class__, model.__class__), {})
        self.__dict__ = model.__dict__
        self.saving_path = saving_path

    def save_model(self, desc: str):
        path = os.path.join(self.saving_path, f'{desc}.pt')
        torch.save(self.state_dict(), path)

    def load_model(self, desc: str = 'best'):
        path = os.path.join(self.saving_path, f'{desc}.pt')
        self.load_state_dict(torch.load(path))

    def from_name(name: str, in_channels: int, num_classes: int, saving_path: str) -> EfficientNet:
        return EfficientNetWrapper(EfficientNet.from_name(name, in_channels=in_channels, num_classes=num_classes),
                                   saving_path=saving_path)
