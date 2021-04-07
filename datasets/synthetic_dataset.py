import torch

from models import ModelFactory

# FIXME: Move to dataset yaml configuration file
GAN_CONFIG = {
    'name': 'wgan',
    'saved_model': 'output/saved_models/WGAN/WGAN_iter_5000',
    'in_dim': (3, 32),
    'args': {
        'generator_iters': 10000
    }
}
LABEL = 1


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, buffer_size: int = 64):
        super(SyntheticDataset).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelFactory.create(GAN_CONFIG).to(self.device)
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
