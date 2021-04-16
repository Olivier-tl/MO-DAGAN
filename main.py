import os
import random
import dataclasses

import fire
import wandb
import torch
import numpy as np

from models import ModelFactory
from datasets import DatasetFactory
from trainers import TrainerFactory
from utils import load_config, logging

logger = logging.getLogger()

OUTPUT_PATH = 'output'
WANDB_TEAM = 'game-theory'
PROJECT_NAME = 'MO-DAGAN'


def main(
        config_path: str = 'configs/classification.yaml',
        dataset_name: str = 'svhn',
        imbalance_ratio: int = 1,
        oversampling: str = 'none',  # none, oversampling, gan
        ada: bool = False,  # only for gan training
        seed: int = 1,  # No seed if 0
        wandb_logs: bool = False,
        test: bool = False,
        load_model: bool = False):
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Set a seed
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Load configuration
    logger.info(f'Loading config at "{config_path}"...')
    config = load_config(config_path, dataset_name, imbalance_ratio, oversampling, ada, load_model)

    if config.trainer.task == 'generation' and test:
        raise ValueError('Cannot test the generation models')

    # Init logging with WandB
    mode = 'offline' if wandb_logs else 'disabled'
    wandb.init(mode=mode,
               dir=OUTPUT_PATH,
               entity=WANDB_TEAM,
               project=PROJECT_NAME,
               group=config.trainer.task,
               config=dataclasses.asdict(config))

    # Load model
    logger.info('Loading model...')
    model = ModelFactory.create(model_config=config.model, n_classes=len(config.dataset.classes))

    # Load dataset
    logger.info('Loading dataset...')
    train_dataset, valid_dataset, test_dataset = DatasetFactory.create(dataset_config=config.dataset, ada=ada)

    # Instatiate trainer
    logger.info('Loading trainer...')
    trainer = TrainerFactory.create(trainer_config=config.trainer,
                                    train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    model=model,
                                    classes=config.dataset.classes)

    if test:
        logger.info('Testing...')
        trainer.test(test_dataset)
    else:
        logger.info('Training...')
        trainer.train()

    # Cleanup
    wandb.finish()

    logger.info('done :)')


if __name__ == '__main__':
    fire.Fire(main)
