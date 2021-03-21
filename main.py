import fire
import yaml

from models import ModelFactory
from datasets import DatasetFactory
from trainers import TrainerFactory
from utils import Config, logging

logger = logging.getLogger()

OUTPUT_PATH = 'output'

def main(config_path: str = 'configs/classification.yaml', dataset_name: str = 'svhn', imbalance_ratio: int = 1):

    # Load configuration
    logger.info(f'Loading config at "{config_path}"...')
    config = Config(config_path=config_path)

    # Load model
    model = ModelFactory.create(model_config=config.model_config)

    # Load dataset
    logger.info('Loading dataset...')
    train_dataset, valid_dataset, _ = DatasetFactory.create(dataset_name=dataset_name,
                                                            imbalance_ratio=imbalance_ratio,
                                                            cache_path=OUTPUT_PATH,
                                                            validation_split=config.validation_split,
                                                            classes=config.classes,
                                                            batch_size=config.batch_size)

    # Instatiate trainer
    trainer = TrainerFactory.create(task=config.task,
                                    train_dataset=train_dataset,
                                    valid_dataset=valid_dataset,
                                    model=model)

    # Train
    logger.info('Training...')
    trainer.train()

    # Test
    logger.info('Testing...')
    trainer.test()

    logger.info('all done :)')


if __name__ == '__main__':
    fire.Fire(main)
