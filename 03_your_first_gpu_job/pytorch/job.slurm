#!/bin/bash
#SBATCH --job-name=svd-torch     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multithread tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module load anaconda3/2020.11
conda activate /scratch/network/$USER/torch-env

python svd_torch.py
