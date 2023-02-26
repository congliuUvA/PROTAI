#!/bin/bash

#SBATCH --job-name=data_generation_3
#SBATCH --tasks=1
#SBATCH --cpus-per-task=46
#SBATCH --mem=64G
#SBATCH --time=7-0
#SBATCH --output=data_gen_3.out

ray start --head
python /home/cliu3/PROTAI/baseline/data_gen.py data.num_partition=4 data.partition_idx=3

