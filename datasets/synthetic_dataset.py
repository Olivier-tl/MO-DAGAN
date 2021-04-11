import os

import torch

from models import ModelFactory
from utils import Config


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_config: Config.Dataset, buffer_size: int = 64, ada: bool = False):
        super(SyntheticDataset).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label = dataset_config.classes[-1]
        ada_suffix = '_ada' if config.trainer.ada else ''
        dataset_config.gan_model.saved_model += f'_{dataset_config.name}_classes_{self.label}{ada_suffix}'
        dataset_config.gan_model.saved_model = os.path.join(dataset_config.gan_model.saved_model, 'final_model')
        self.model = ModelFactory.create(dataset_config.gan_model).to(self.device)

        self.buffer_size = buffer_size
        self.pointer = 0
        self.buffer = self.generate_buffer()

    def generate_buffer(self):
        z = torch.randn(self.buffer_size, 100, 1, 1).to(self.device)
        samples = self.model.G(z)
        samples = samples.mul(0.5).add(0.5)
        return samples.detach().cpu()  # Move on cpu because other datasets are on cpu before being copied to gpu

    def __iter__(self):
        while True:
            if self.pointer >= self.buffer_size:
                self.buffer = self.generate_buffer()
                self.pointer = 0
            sample = self.buffer[self.pointer]
            self.pointer += 1
            yield (sample, self.label)
