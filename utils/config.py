import pprint
import yaml

from utils import logging

logger = logging.getLogger()


class Config:
    def __init__(self, config_path: str):
        self.config = yaml.load(open(config_path), Loader=yaml.Loader)
        self.task = self.config['task']
        self.model_config = self.config['model']
        self.validation_split = self.config['dataset']['validation_split']
        self.classes = self.config['dataset']['classes']
        self.batch_size = self.config['dataset']['batch_size']

    def print(self):
        pprint.pprint(self.config)
