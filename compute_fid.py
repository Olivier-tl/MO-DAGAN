import subprocess
import os

import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from utils import load_config
from datasets import DatasetFactory
from datasets import SyntheticDataset


def compute_fid(config_path='configs/classification.yaml',
                dataset_name='cifar10',
                imbalance_ratio=100,
                oversampling='none',  # none, oversampling, gan
                ada=False,
                load_model=False):

    config_gen = load_config(config_path, dataset_name, imbalance_ratio, oversampling, ada, load_model)
    config_gen.dataset.classes = [1]

    # Instantiating dataloaders
    dataloader_true, _, _ = DatasetFactory.create(dataset_config=config_gen.dataset)
    dataloader_gen = SyntheticDataset(config_gen.dataset)

    true_path = f'output/fid_samples_true/{dataset_name}_{imbalance_ratio}_ada_{ada}'
    gen_path = f'output/fid_samples_gen/{dataset_name}_{imbalance_ratio}_ada_{ada}'
    # if not (os.path.exists(true_path) and os.path.exists(gen_path)):
    os.makedirs(true_path, exist_ok=True)
    os.makedirs(gen_path, exist_ok=True)

    # Saving real images to folder
    cpt = 0
    for batch in tqdm(dataloader_true, desc='saving real img'):
        for img in batch[0]:
            save_image(img, os.path.join(true_path, f'img_{cpt}.png'))
            cpt += 1

    # Saving generated images to folder
    for i in tqdm(range(cpt), desc='saving fake img'):
        img = next(iter(dataloader_gen))[0]
        save_image(img, os.path.join(gen_path, 'img_' + str(i) + ".png"))

    # Computing FID
    print('computing fid...')
    completed_process = subprocess.run(["python", "-m", "pytorch_fid", true_path, gen_path], capture_output=True, text=True)
    fid = float(completed_process.stdout.split(' ')[-1])
    output = (f'{dataset_name}_{imbalance_ratio}_ada_{ada}_', fid)
    print(output)
    return output

if __name__ == '__main__':
    dataset_names = ['mnist', 'fashion-mnist', 'svhn', 'cifar10']
    imbalance_ratios = [10, 50, 100]
    adas = [True, False]
    outputs = []

    cpt = 0
    total = len(dataset_names) * len(imbalance_ratios) * len(adas)
    for dataset_name in dataset_names:
        for imbalance_ratio in imbalance_ratios:
            for ada in adas:
                cpt += 1
                print(f'run {cpt} / {total}')
                outputs.append(compute_fid(dataset_name=dataset_name, imbalance_ratio=imbalance_ratio, ada=ada))
    print('----- DONE -----')
    print(outputs)

