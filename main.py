import random

import fire
import yaml
import torch
import numpy as np

from models import ModelFactory
from datasets import DatasetFactory
from trainers import TrainerFactory
from utils import Config, logging

logger = logging.getLogger()

OUTPUT_PATH = 'output'
DATASET_INPUT_SIZES = { 
                        # dataset_name: (channels, img_size (assumed square))
                        'mnist':(1, 28),
                        'fashion-mnist':(1, 28),
                        'cifar10':(3, 32),
                        'svhn':(3, 32), 
                        }


def main(
        config_path: str = 'configs/classification.yaml',
        dataset_name: str = 'svhn',
        imbalance_ratio: int = 1,
        seed: int = 1,  # No seed if 0
):

    # Setting a seed
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Load configuration
    logger.info(f'Loading config at "{config_path}"...')
    config = Config(config_path=config_path)

    # Load model
    logger.info('Loading model...')
    model_config = config.model_config
    model_config['in_dim'] = DATASET_INPUT_SIZES[dataset_name]
    model = ModelFactory.create(model_config=model_config)

    # Load dataset
    logger.info('Loading dataset...')
    train_dataset, valid_dataset, _ = DatasetFactory.create(dataset_name=dataset_name,
                                                            imbalance_ratio=imbalance_ratio,
                                                            cache_path=OUTPUT_PATH,
                                                            validation_split=config.validation_split,
                                                            classes=config.classes,
                                                            batch_size=config.batch_size)

    # Instatiate trainer
    logger.info('Loading trainer...')
    trainer = TrainerFactory.create(task=config.task,
                                    train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    model=model)

    # Train
    logger.info('Training...')
    trainer.train()

    logger.info('all done :)')


if __name__ == '__main__':
    fire.Fire(main)
