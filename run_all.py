import os
import subprocess

import fire

datasets = ['mnist', 'fashion-mnist', 'cifar10', 'svhn']
oversampling_options = ['none', 'oversampling', 'gan']
imbalance_ratios = [10, 50, 100]
ada_options = [True, False]


def run_classification():
    for dataset in datasets:
        for oversampling in oversampling_options:
            for imbalance_ratio in imbalance_ratios:
                for ada in ada_options:
                    if oversampling != 'gan' and ada:
                        continue
                    subprocess.run([
                        'python', 'main.py', '--config_path=configs/classification.yaml', f'--dataset_name={dataset}',
                        f'--oversampling={oversampling}', f'--imbalance_ratio={imbalance_ratio}', f'--ada={ada}',
                        '--wandb_logs=True'
                    ])


def run_generation():
    for dataset in datasets:
        for ada in ada_options:
            subprocess.run([
                'python', 'main.py', '--config_path=configs/gan.yaml', f'--dataset_name={dataset}', f'--ada={ada}',
                '--imbalance_ratio=1', '--oversampling=none', '--wandb_logs=True'
            ])


def main(task: str = 'classification'):
    if task == 'classification':
        run_classification()
    elif task == 'generation':
        run_generation()
    else:
        raise ValueError(f'Task {task} not recognized.')

    print('all done ;}')


if __name__ == '__main__':
    fire.Fire(main)