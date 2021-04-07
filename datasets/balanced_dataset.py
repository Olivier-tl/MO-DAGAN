from collections import defaultdict
from collections import Counter

import torch


class BalancedDataset(torch.utils.data.IterableDataset):
    def __init__(self, unbalanced_dataset: torch.utils.data.Dataset, minority_dataset: torch.utils.data.Dataset):
        self.unbalanced_dataset = unbalanced_dataset
        self.minority_dataset = minority_dataset
        self.class_count = self.get_class_count(self.unbalanced_dataset)

        self.num_new_sample = max(self.class_count.values()) - min(self.class_count.values())
        print('number of new samples : ', self.num_new_sample)
        print('total number of examples: ', len(self.unbalanced_dataset))
        print('number of example in the minority class : ', min(self.class_count.values()))
        print('number of example in the majority class : ', max(self.class_count.values()))

        self.ratio = self.num_new_sample / (self.num_new_sample + len(self.unbalanced_dataset))
        print('ratio : ', self.ratio)
        self.counter = 0.0

    def get_class_count(self, dataset):
        class_count = defaultdict(int)
        for data, label in dataset:
            class_count[label] += 1
        return class_count

    def __iter__(self):

        for data in self.unbalanced_dataset:
            self.counter += self.ratio
            if (self.counter >= 1):
                self.counter = self.counter % 1
                yield next(iter(self.minority_dataset))
                self.counter += self.ratio
            yield data

    def __len__(self):
        return len(self.unbalanced_dataset) + self.num_new_sample
