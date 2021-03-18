from .trainer import Trainer

# TODO: Implement the classification trainer (issue #4)
class ClassificationTrainer(Trainer):
    def __init__(model: torch.nn.module, dataset: torch.utils.data.DataLoader):
        self.model = model
        self.dataset = dataset

    def train():
        pass

    def test():
        pass