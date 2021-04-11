import os
import subprocess

import fire

datasets = ['mnist', 'fashion-mnist', 'cifar10', 'svhn']
oversampling_options = ['none', 'oversampling', 'gan']  # TODO: Add gan+ada
imbalance_ratios = [10, 50, 100]


def run_classification():
    for dataset in datasets:
        for oversampling in oversampling_options:
            for imbalance_ratio in imbalance_ratios:
                subprocess.run([
                    'python', 'main.py', '--config_path=configs/classification.yaml', f'--dataset_name={dataset}',
                    f'--oversampling={oversampling}', f'--imbalance_ratio={imbalance_ratio}', '--wandb_logs=True'
                ])


def run_generation():
    for dataset in datasets:
        subprocess.run([
            'python', 'main.py', '--config_path=configs/gan.yaml', f'--dataset_name={dataset}', f'--imbalance_ratio=1',
            f'--oversampling=none', f'--wandb_logs=True'
        ])


def main(task: str = 'classification'):
    if task == 'classification':
        run_classification()
    elif task == 'generation':
        run_generation()
    else:
        raise ValueError(f'Task {task} not recognized.')

    print('all done :)')


if __name__ == '__main__':
    fire.Fire(main)