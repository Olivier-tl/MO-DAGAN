import torch


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, model: torch.nn.Module, label: str, buffer_size: int = 64):
        super(SyntheticDataset).__init__()
        self.model = model
        self.label = label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.pointer = 0
        self.buffer = self.generate_buffer()

    def generate_buffer(self):
        z = torch.randn(self.buffer_size, 100, 1, 1).to(self.device)
        samples = self.model.G(z)
        samples = samples.mul(0.5).add(0.5)
        return samples

    def __iter__(self):
        while True:
            if self.pointer >= self.buffer_size:
                self.buffer = self.generate_buffer()
                self.pointer = 0
            sample = self.buffer[self.pointer]
            self.pointer += 1
            yield (sample, self.label)
