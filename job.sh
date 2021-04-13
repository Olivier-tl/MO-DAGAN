#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:0:0
#SBATCH --mem=8G
#SBATCH --account=def-bengioy

source venv/bin/activate
cd MO-DAGAN/
python run_all.py --task=generation
python run_all.py --task=classification