import itertools
import typing
import random
from collections import defaultdict

import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Artificially samples classes from a multiclass dataset and create an imbalance according 
    to the specified imbalance ratio (IR = n_first_class/n_last_class).
    """
    def __init__(self, dataset: torch.utils.data.dataset, imbalance_ratio: int, classes: typing.List[int]):
        """Constructor

        Args:
            dataset (torch.utils.data.dataset): Original balanced dataset.
            imbalance_ratio (int): Imbalance ratio (n_first_class/n_last_class).
            classes (typing.List[int]): List of classes to sample from.
        """
        if imbalance_ratio != 1 and len(classes) == 1:
            raise ValueError("Can't create an imbalance if only one classe is selected.")

        class_idx = defaultdict(list)
        for idx in list(range(len(dataset))):
            label = dataset[idx][1]
            for i in range(len(classes)):
                if label == classes[i]:
                    class_idx[label].append(idx)

        trunc_idx = int(len(class_idx[classes[0]]) // imbalance_ratio)
        random.shuffle(class_idx[classes[-1]])
        class_idx[classes[-1]] = class_idx[classes[-1]][:trunc_idx]
        self.imbalanced_indices = list(
            itertools.chain.from_iterable([class_idx[classes[i]] for i in range(len(classes))]))

    def __iter__(self):
        return (self.imbalanced_indices[i] for i in torch.randperm(len(self.imbalanced_indices)))

    def __len__(self):
        return len(self.imbalanced_indices)
