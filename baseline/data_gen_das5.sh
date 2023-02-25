#!/bin/bash

#SBATCH --job-name=data_generation_5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-0
#SBATCH -o data_gen_5.out

ray start --head
python /var/scratch/cliu4/PROTAI/baseline/data_gen.py data.num_partition=6 data.partition_idx=5

