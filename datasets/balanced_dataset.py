from collections import defaultdict
from collections import Counter

import torch


class BalancedDataset(torch.utils.data.IterableDataset):
    def __init__(self, unbalanced_dataset: torch.utils.data.Dataset, minority_dataset: torch.utils.data.Dataset):
        self.unbalanced_dataset = unbalanced_dataset
        self.minority_dataset = minority_dataset
        self.class_count = self.get_class_count(self.unbalanced_dataset)

        self.num_new_sample = max(list(self.class_count.values())) - min(list(self.class_count.values()))
        self.ratio = self.num_new_sample / (self.num_new_sample + len(self.unbalanced_dataset))
        self.counter = 0.0

    def get_class_count(self, dataset):
        class_count = defaultdict(int)
        for idx in list(range(len(dataset))):
            label = dataset[idx][1]
            class_count[label] += 1
        return class_count

    def __iter__(self):
        for data in self.unbalanced_dataset:
            self.counter += self.ratio
            if (self.counter >= 1):
                self.counter = self.counter % 1
                yield next(iter(self.minority_dataset))
            yield data

    def __len__(self):
        return len(self.unbalanced_dataset) + self.num_new_sample
