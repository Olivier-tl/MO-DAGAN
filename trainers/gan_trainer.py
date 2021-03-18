from .trainer import Trainer

# TODO: Implement the gan trainer (issue #7)
class GANTrainer(Trainer):
   def __init__(model: torch.nn.module, dataset: torch.utils.data.DataLoader):
        self.model = model
        self.dataset = dataset

    def train():
        pass

    def test():
        pass