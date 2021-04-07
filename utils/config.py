import typing
import yaml
from dataclasses import dataclass

from marshmallow_dataclass import class_schema

DATASET_INPUT_SIZES = {
    # dataset_name: (channels, img_size (assumed square))
    'mnist': (1, 28),
    'fashion-mnist': (1, 28),
    'cifar10': (3, 32),
    'svhn': (3, 32),
}


def load_config(path: str, dataset_name: str, imbalance_ratio: int, oversampling: str):
    config_schema = class_schema(Config)
    config_dict = yaml.load(open(path), Loader=yaml.Loader)
    config = config_schema().load(config_dict)
    config.dataset.name = dataset_name
    config.dataset.imbalance_ratio = imbalance_ratio
    config.dataset.oversampling = oversampling
    config.model.input_channels = DATASET_INPUT_SIZES[dataset_name][0]
    config.model.output_size = DATASET_INPUT_SIZES[dataset_name][1]
    return config


@dataclass
class Config:
    @dataclass
    class Model:
        name: str = None
        saved_model: str = None
        input_channels: int = None
        output_size: int = None  # assumed square (eg: 32 -> 32x32)

    @dataclass
    class Trainer:
        task: str = None  # classification, generation
        lr: float = None
        optimizer: str = None
        loss: str = None
        epochs: int = None

    @dataclass
    class Dataset:
        name: str = None
        validation_split: float = None
        classes: typing.List[int] = None
        oversampling: str = None  # none, oversampling, gan
        gan_model: typing.Any = None  # model config if oversampling is gan
        imbalance_ratio: int = None  # IR = n_maj_class / n_min_class
        batch_size: int = None

    model: Model
    trainer: Trainer
    dataset: Dataset
