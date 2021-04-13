import typing
import yaml
from dataclasses import dataclass

from marshmallow_dataclass import class_schema

DATASET_NB_CHANNELS = {
    'mnist': 1,
    'fashion-mnist': 1,
    'cifar10': 3,
    'svhn': 3,
}


def load_config(path: str, dataset_name: str, imbalance_ratio: int, oversampling: str, ada: bool):
    config_schema = class_schema(Config)
    config_dict = yaml.load(open(path), Loader=yaml.Loader)
    config = config_schema().load(config_dict)
    config.trainer.ada = ada
    config.dataset.name = dataset_name
    config.dataset.imbalance_ratio = imbalance_ratio
    config.dataset.oversampling = oversampling
    config.model.input_channels = DATASET_NB_CHANNELS[dataset_name]
    config.dataset.gan_model.input_channels = DATASET_NB_CHANNELS[dataset_name]
    return config


@dataclass
class Config:
    @dataclass
    class Model:
        name: str = None
        saved_model: str = None
        input_channels: int = None

    @dataclass
    class Trainer:
        task: str = None  # classification, generation
        lr: float = None
        optimizer: str = None
        loss: str = None
        epochs: int = None
        ada: bool = None
        betas: typing.List[float] = None

    @dataclass
    class Dataset:

        # FIXME: Ideally we would reuse the Model class above but
        #        it is not defined inside the Dataset class
        @dataclass
        class Model:
            name: str = None
            saved_model: str = None
            input_channels: int = None

        name: str = None
        validation_split: float = None
        classes: typing.List[int] = None
        oversampling: str = None  # none, oversampling, gan
        gan_model: Model = None  # model config if oversampling is gan
        imbalance_ratio: int = None  # IR = n_maj_class / n_min_class
        batch_size: int = None

    model: Model
    trainer: Trainer
    dataset: Dataset
