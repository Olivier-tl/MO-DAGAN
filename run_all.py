import os
import subprocess

import fire

datasets = ['mnist', 'fashion-mnist', 'cifar10', 'svhn']  # FIXME: Missing mnist
oversampling_options = ['none', 'oversampling', 'gan']
imbalance_ratios = [10, 50, 100]
ada_options = [False, True]


def run_classification(test: bool):
    for dataset in datasets:
        for oversampling in oversampling_options:
            for imbalance_ratio in imbalance_ratios:
                for ada in ada_options:
                    if oversampling != 'gan' and ada:
                        continue
                    command = [
                        'python', 'main.py', '--config_path=configs/classification.yaml',
                        f'--dataset_name={dataset}', f'--oversampling={oversampling}',
                        f'--imbalance_ratio={imbalance_ratio}', f'--ada={ada}',
                        f'--test={test}', f'--load_model={test}', '--wandb_logs=True'
                    ]
                    print(command)
                    subprocess.run(command)


def run_generation():
    for dataset in datasets:
        for imbalance_ratio in imbalance_ratios:
            for ada in ada_options:
                command = [
                    'python', 'main.py', '--config_path=configs/gan.yaml',
                    f'--dataset_name={dataset}', f'--ada={ada}',
                    f'--imbalance_ratio={imbalance_ratio}', '--oversampling=none',
                    '--wandb_logs=True'
                ]
                print(command)
                subprocess.run(command)


def main(task: str = 'classification', test: bool = False):
    if task == 'classification':
        run_classification(test)
    elif task == 'generation':
        run_generation()
    else:
        raise ValueError(f'Task {task} not recognized.')

    print('all done ;}')


if __name__ == '__main__':
    fire.Fire(main)