import os
import random

import fire
import wandb
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
WANDB_TEAM = 'game-theory'
PROJECT_NAME = 'MO-DAGAN'


def main(
    config_path: str = 'configs/classification.yaml',
    dataset_name: str = 'svhn',
    #imbalance_ratio: int = 1,
    imbalance_ratio: int = 10,  # DELEETE ME
    seed: int = 1,  # No seed if 0
    wandb_logs: bool = True,
):
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
    config = Config(config_path=config_path)

    # Init logging with WandB
    mode = 'offline' if wandb_logs else 'disabled'
    wandb.init(mode=mode,
               dir=OUTPUT_PATH,
               entity=WANDB_TEAM,
               project=PROJECT_NAME,
               group=config.task,
               config=config.config)
    wandb_config = wandb.config
    wandb_config.dataset_name = dataset_name
    wandb_config.imbalance_ratio = imbalance_ratio

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

    # delete after test !!!!!
    #model.load_model("output/saved_models/WGAN/WGAN_iter_5000")
    config2 = Config(config_path="configs/gan.yaml")
    model = ModelFactory.create(model_config=config2.model_config)
    model.to(torch.device("cuda"))
    syn_test_ds = SyntheticDataset(model, 1)
    print("allo")
    #for exemple in (iter(syn_test_ds)):
    #    print(exemple[0][0][3,3])
    from collections import Counter
    #print(dict(Counter((train_dataset.dataset).targets)))
    bal_test_ds = BalancedDataset(train_dataset.dataset, syn_test_ds)
    print(bal_test_ds.get_class_count(train_dataset.dataset))
    print(bal_test_ds.get_class_count(bal_test_ds))
    #print(dict(Counter(bal_test_ds.targets)))

    quit()

    # Instatiate trainer
    logger.info('Loading trainer...')
    trainer = TrainerFactory.create(task=config.task,
                                    train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    model=model)

    # Train
    logger.info('Training...')
    trainer.train()

    # Cleanup
    wandb.finish()

    logger.info('all done :)')


if __name__ == '__main__':
    fire.Fire(main)
