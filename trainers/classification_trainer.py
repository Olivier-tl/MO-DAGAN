import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .trainer import Trainer

# Loosely based on : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# Receives: an untrained, but constructed model/net
#           a dataset
#           a seed
# Needs to : load the dataset
#            normalize the dataset
#            split the dataset
#            define a loss function
#            train the untrained net using the normalized and splitted train data
#            test the trained net using the normalized and splitted test data

class ClassificationTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, train_dataset: torch.utils.data.DataLoader, valid_dataset: torch.utils.data.DataLoader, lr = 0.001, optimizer = "Adam", loss = "cross entropy", num_epoch = 2, seed = 0):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.optimizer = self.getOptimizer(optimizer)
        self.loss = self.getLoss(loss)
        self.num_epoch = num_epoch
        if seed :
            #torch.manual_seed(seed)
            #torch.cuda.manual_seed_all(seed)
            #np.random.seed(seed)
            #random.seed(seed)
            #torch.backends.cudnn.deterministic=True
            pass

    # Might need to set more parameters
    def getOptimizer(self, opt):
        if opt == "Adam":
            return optim.Adam(self.model.parameters(), self.lr)
        elif opt == "SGD":
            return optim.SGD(self.model.parameters(), self.lr)
        else :
            print("For now, use Adam or SGD optimizers. Falling back to Adam by default")
            return optim.Adam(self.model.parameters(), self.lr)

    # Might need to set more possible losses/criterions
    def getLoss(self, opt):
        if opt == "cross entropy":
            return nn.CrossEntropyLoss()
        else :
            print("For now, use cross entropy loss. Falling back to cross entropy loss by default")
            return nn.crossEntropyLoss()

    def save_model(self, name):
        PATH = '../saved_models/' + name
        torch.save(self.model.state_dict(), PATH)

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
                    print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / iter_run_loss))
                    running_loss = 0.0

        print('Finished Training')

    def test(self):

        self.model.eval()
        with torch.no_grad():

            running_loss = 0.0
            for i, data in enumerate(self.valid_dataset, 0):

                inputs, labels = data
                outputs = self.model(inputs)

                loss = self.loss(outputs, labels)

                running_loss+= loss.item()

        print("Valid set avg loss : %.5f" %(running_loss / len(self.valid_dataset.dataset)))
