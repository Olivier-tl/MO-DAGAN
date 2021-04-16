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

        minority_class_only = False
        if imbalance_ratio != 1 and len(classes) == 1:
            majority_class = classes[0] - 1
            if majority_class < 0:
                raise ValueError('Unknown majority class with which to create imbalance')
            classes.insert(0, majority_class)
            minority_class_only = True

        class_idx = defaultdict(list)
        self.labels = []
        for idx in list(range(len(dataset))):
            label = dataset[idx][1]
            for i in range(len(classes)):
                if label == classes[i]:
                    class_idx[label].append(idx)

        trunc_idx = int(len(class_idx[classes[0]]) // imbalance_ratio)
        random.shuffle(class_idx[classes[-1]])
        class_idx[classes[-1]] = class_idx[classes[-1]][:trunc_idx]

        if minority_class_only:
            classes = [classes[-1]]
        self.imbalanced_indices = list(
            itertools.chain.from_iterable([class_idx[classes[i]] for i in range(len(classes))]))

        # Compute labels
        self.labels = torch.tensor([dataset[idx][1] for idx in self.imbalanced_indices])

    def __getitem__(self, idx):
        return self.dataset[self.imbalanced_indices[idx]]

    def __len__(self):
        return len(self.imbalanced_indices)