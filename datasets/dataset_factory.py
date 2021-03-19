import torch

from .imbalanced_dataset_sampler import ImbalancedDatasetSampler

class DatasetFactory:
    def create(dataset_name: str, imbalance_ratio: int) -> torch.utils.data.DataLoader:

        # TODO: Create proper dataloader with proper sampler 
        if dataset_name == 'mnist':
            dataset = None
        elif dataset_name == 'fashion-mnist':
            dataset = None
        elif dataset_name == 'cifar10':
            dataset = None
        elif dataset_name == 'svhn':
            dataset = None
        return dataset