import torch
from efficientnet_pytorch import EfficientNet


class EfficientNetWrapper(EfficientNet):
    """Light wrapper around EfficientNet to add model saving/loading.
    """
    def __init__(self, model: EfficientNet):
        self.__class__ = type(model.__class__.__name__, (self.__class__, model.__class__), {})
        self.__dict__ = model.__dict__

    def save_model(self, path: str):
        torch.save(self.state_dict(), f'{path}.pt')

    def load_model(self, path: str):
        self.load_state_dict(torch.load(f'{path}.pt'))

    def from_name(name: str, in_channels: int, image_size: int) -> EfficientNet:
        return EfficientNetWrapper(EfficientNet.from_name(name, in_channels=in_channels, image_size=image_size))
