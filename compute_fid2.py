import subprocess
import os

import numpy as np
import torch
from torchvision.utils import save_image

from utils import load_config
from datasets import DatasetFactory
from datasets import SyntheticDataset



config_path = 'configs/classification.yaml'
dataset_name = 'cifar10'
imbalance_ratio = 100
oversampling = 'none'  # none, oversampling, gan
ada = False
load_model: bool = False

config_gen = load_config(config_path, dataset_name, imbalance_ratio, oversampling, ada, load_model)
config_gen.dataset.classes = [1]
sub_ds_size = config_gen.dataset.batch_size

# Instantiating dataloaders
dataloader_true, _, _ = DatasetFactory.create(dataset_config=config_gen.dataset)
dataloader_gen = SyntheticDataset(config_gen.dataset)

# Saving DS images to folder
os.makedirs('output/fid_samples_true/', exist_ok=True)
os.makedirs('output/fid_samples_gen/', exist_ok=True)
imgs= next(iter(dataloader_true))
for i,im in enumerate(imgs[0]):
    save_image(im, "output/fid_samples_true/im" + str(i) + ".png")

# Saving generated images to folder
for i in range(sub_ds_size):
    im = next(iter(dataloader_gen))[0]
    save_image(im, "output/fid_samples_gen/im" + str(i) + ".png")

# Computing FID
subprocess.run(["python", "-m", "pytorch_fid", "output/fid_samples_true", "output/fid_samples_gen"])
