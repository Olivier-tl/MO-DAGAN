import itertools
import typing
import random
from collections import defaultdict

import torch


class ImbalancedDataset(torch.utils.data.Dataset):
    """
    Creates a new dataset by sampling classes from a multiclass dataset and create an imbalance according 
    to the specified imbalance ratio (IR = n_first_class/n_last_class).
    """
    def __init__(self, dataset: torch.utils.data.dataset, imbalance_ratio: int, classes: typing.List[int]):
        """Constructor

        Args:
            dataset (torch.utils.data.dataset): Original balanced dataset.
            imbalance_ratio (int): Imbalance ratio (n_first_class/n_last_class).
            classes (typing.List[int]): List of classes to sample from.
        """
        self.dataset = dataset

        if imbalance_ratio != 1 and len(classes) == 1:
            raise ValueError("Can't create an imbalance if only one classe is selected.")

        class_idx = defaultdict(list)
        self.labels = []
        for idx in list(range(len(dataset))):
            label = dataset[idx][1]
            self.labels.append(label)
            for i in range(len(classes)):
                if label == classes[i]:
                    class_idx[label].append(idx)
        self.labels = torch.tensor(self.labels)

        trunc_idx = int(len(class_idx[classes[0]]) // imbalance_ratio)
        random.shuffle(class_idx[classes[-1]])
        class_idx[classes[-1]] = class_idx[classes[-1]][:trunc_idx]
        self.imbalanced_indices = list(
            itertools.chain.from_iterable([class_idx[classes[i]] for i in range(len(classes))]))

    def __getitem__(self, idx):
        return self.dataset[self.imbalanced_indices[idx]]

    def __len__(self):
        return len(self.imbalanced_indices)