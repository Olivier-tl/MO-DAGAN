import yaml

from utils import logging

logger = logging.getLogger()


class Config:
    def __init__(self, config_path: str):
        self.config = yaml.load(open(config_path), Loader=yaml.Loader)
        self.task = self.config['task']
        self.model_name = self.config['model']['name']
        self.validation_split = self.config['dataset']['validation_split']
        self.classes = self.config['dataset']['classes']

        # Add more config attributes here
        # self.new_attribute = ...

    def print(self):
        logger.info(f'Config: {self.config}')
