#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=30:0:0
#SBATCH --mem=8G
#SBATCH --account=def-bengioy
#SBATCH --mail-user=oliviertl@hotmail.com
#SBATCH --mail-type=ALL

source ../venv/bin/activate
python run_all.py --task=generation
# python run_all.py --task=classification --test=True