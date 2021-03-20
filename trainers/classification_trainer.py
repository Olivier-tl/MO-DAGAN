import torch
import torch.optim as optim
import torch.nn as nn

from .trainer import Trainer

# Loosely based on : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

# TODO: Implement the classification trainer (issue #4)

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
    def __init__(self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.DataLoader,
        valid_dataset: torch.utils.data.DataLoader,
        optimizer = "Adam",
        loss = "cross entropy",
        num_epoch = 1,
        seed = None
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.optimizer = getOptimizer(optimizer)
        self.loss = getLoss(loss)
        if seed :
            torch.manual_seed(seed)

    # Might need to set more parameters
    def getOptimizer(self, opt):
        lr = 0.0001
        if opt == "Adam":
            return optim.Adam(self.model.parameters(), lr)
        elif opt == "SGD":
            return optim.SGD(self.model.parameters(), lr)
        else :
            print("For now, use Adam or SGD optimizers. Falling back to Adam by default")
            return optim.Adam(self.model.parameters(), lr)

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
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def test(self):

        model.eval()
        with torch.no_grad():

            running_loss = 0.0
            for i, data in enumerate(self.valid_dataset, 0):

                inputs, labels = data
                outputs = self.model(inputs)

                loss = self.loss(outputs, labels)

                running_loss+= loss.item()

            print("Valid set avg loss : %.3f" %(running_loss / len(self.valid_dataset.dataset)))
