import torch

from models import ModelFactory
from utils import Config

# FIXME: Move to dataset yaml configuration file
LABEL = 1


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, model_config: Config.Model, buffer_size: int = 64):
        super(SyntheticDataset).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelFactory.create(model_config).to(self.device)
        self.label = LABEL

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
