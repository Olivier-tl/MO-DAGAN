import os
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


def load_config(path: str, dataset_name: str, imbalance_ratio: int, oversampling: str, ada: bool, load_model: bool):

    # Load config from yaml file
    config_schema = class_schema(Config)
    config_dict = yaml.load(open(path), Loader=yaml.Loader)
    config = config_schema().load(config_dict)

    # Set config from console arguments
    config.trainer.ada = ada
    config.dataset.name = dataset_name
    config.dataset.imbalance_ratio = imbalance_ratio
    config.dataset.oversampling = oversampling
    config.dataset.gan_model.input_channels = DATASET_NB_CHANNELS[dataset_name]
    config.model.input_channels = DATASET_NB_CHANNELS[dataset_name]
    config.model.load = load_model

    # Dynamically set the saving path based on config
    oversampling_suffix = f'_oversampling-{oversampling}' if config.trainer.task == 'classification' else ''
    imbalance_ratio_suffix = f'_IR-{imbalance_ratio}'
    ada_suffix = '_ada' if config.trainer.ada else ''
    config.model.saved_model += f'_{dataset_name}{oversampling_suffix}{imbalance_ratio_suffix}_classes_{"-".join(map(str, config.dataset.classes))}{ada_suffix}'
<<<<<<< HEAD
    config.dataset.gan_model.saved_model += f'_{dataset_name}{imbalance_ratio_suffix}_classes_{config.dataset.classes[-1]}{ada_suffix}'
=======
>>>>>>> 1d6922fb87c162eb5d20967475fcee69a7cb7760
    os.makedirs(config.model.saved_model, exist_ok=True)
    return config


@dataclass
class Config:
    @dataclass
    class Model:
        name: str = None
        load: bool = None
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
            load: bool = True
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
