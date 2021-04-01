# Loosely based on : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils import logging
from .trainer import Trainer

logger = logging.getLogger()


class ClassificationTrainer(Trainer):
    """Trainer for classification models which provide methods to train 
    an untrained net using splitted train data and test the trained net 
    using the splitted test data. 
    """
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataset: DataLoader,
                 valid_dataset: DataLoader,
                 lr: float = 0.001,
                 optimizer: str = "adam",
                 loss: str = "cross_entropy",
                 num_epoch: int = 2):
        super(ClassificationTrainer, self).__init__(model, lr, optimizer, loss, num_epoch)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.num_epoch = num_epoch

    def train(self):

        for epoch in range(self.num_epoch):

            running_loss = 0.0
            for i, data in enumerate(self.train_dataset, 0):

                print("batch : ", i + 1)

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)

                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                #if i % 2000 == 1999:    # print every 2000 mini-batches
                iter_run_loss = 10
                if i % iter_run_loss == iter_run_loss - 1:
                    logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / iter_run_loss))
                    running_loss = 0.0

        logger.info('Finished Training')

    def test(self):

        self.model.eval()
        with torch.no_grad():

            running_loss = 0.0
            for i, data in enumerate(self.valid_dataset, 0):

                inputs, labels = data
                outputs = self.model(inputs)

                loss = self.loss(outputs, labels)

                running_loss += loss.item()

        logger.info('Valid set avg loss : %.5f' % (running_loss / len(self.valid_dataset.dataset)))
