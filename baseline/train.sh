#!/bin/bash

#SBATCH --job-name=train
#SBATCH --tasks=1
#SBATCH --cpus-per-task=46
#SBATCH --gres=gpu:8
#SBATCH --mem=120G
#SBATCH --time=7-0

python /home/cliu3/PROTAI/baseline/train.py >train.out

