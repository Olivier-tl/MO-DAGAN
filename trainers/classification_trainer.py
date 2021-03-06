# Loosely based on : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import random
from typing import Tuple, List

import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils import logging, metrics, Config
from .trainer import Trainer

logger = logging.getLogger()


class ClassificationTrainer(Trainer):
    """Trainer for classification models which provide methods to train 
    an untrained net using splitted train data and test the trained net 
    using the splitted test data. 
    """
    def __init__(self, trainer_config: Config.Trainer, model: torch.nn.Module, train_dataset: DataLoader,
                 valid_dataset: DataLoader, classes: List[int]):
        super(ClassificationTrainer, self).__init__(model, classes)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = self._get_optimizer(trainer_config.optimizer, model, trainer_config.lr, trainer_config.betas)
        self.loss = self._get_loss(trainer_config.loss)
        self.num_epoch = trainer_config.epochs

    def train(self):
        best_accuracy = 0
        for epoch in range(self.num_epoch):
            self.model.train()
            total_loss = 0.0
            total_accuracy = 0.0
            total_examples = 0
            with tqdm.tqdm(enumerate(self.train_dataset, 0),
                           desc=f'Training Epoch {epoch+1}',
                           total=len(self.train_dataset)) as train_pbar:
                for i, data in train_pbar:

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs)

                    loss = self.loss(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # compute accuracy
                    _, preds = torch.max(outputs.data, dim=1)
                    accuracy = (preds == labels).sum().float()

                    # print statistics
                    train_pbar.set_postfix({'loss': f'{loss.item()/len(labels):.3f}', 'accuracy': f'{accuracy.item()/len(labels):.3f}'})
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    total_examples += len(inputs)

                total_loss /= total_examples
                total_accuracy /= total_examples

                # log metrics
                train_pbar.set_postfix({'loss': f'{total_loss:.3f}', 'accuracy': f'{total_accuracy:.3f}'})
                wandb.log({'train_loss': total_loss, 'train_accuracy': total_accuracy}, commit=False)

            # validation
            valid_loss, valid_accuracy = self.test(self.valid_dataset, desc='valid')
            if valid_accuracy > best_accuracy:
                self.save_model(desc=f'best')
                best_accuracy = valid_accuracy

        logger.info('Finished Training')

    def test(self, test_dataset: DataLoader, desc: str = 'test') -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            with tqdm.tqdm(enumerate(test_dataset, 0), desc=desc, total=len(test_dataset)) as test_pbar:
                confusion_matrix = None
                total_loss = 0.0
                total_accuracy = 0.0
                total_examples = 0
                for i, data in test_pbar:

                    inputs, labels = data[0].to(self.device), data[1].to(self.device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs.data, dim=1)

                    loss = self.loss(outputs, labels)
                    accuracy = (preds == labels).sum().float()
                    if confusion_matrix == None:
                        confusion_matrix = metrics.get_confusion_matrix(preds, labels, len(self.classes))
                    else:
                        confusion_matrix += metrics.get_confusion_matrix(preds, labels, len(self.classes))

                    test_pbar.set_postfix({'loss': f'{loss.item()/len(labels):.3f}', 'accuracy': f'{accuracy.item()/len(labels):.3f}'})
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    total_examples += len(inputs)

                total_loss /= total_examples
                total_accuracy /= total_examples
                total_csa = metrics.get_CSA(confusion_matrix)
                total_acsa = torch.mean(total_csa)
                test_pbar.set_postfix({'loss': f'{total_loss:.3f}', 'accuracy': f'{total_accuracy:.3f}'})
                csa_keys = [f'Class {c} accuracy' for c in self.classes]
                csa_dict = dict(zip(csa_keys, total_csa.cpu().numpy()))
                wandb.log(csa_dict, commit=False)
                wandb.log({
                    f'{desc}_loss': total_loss,
                    f'{desc}_accuracy': total_accuracy,
                    f'{desc}_acsa': total_acsa
                },
                          commit=True)
        return total_loss, total_accuracy
