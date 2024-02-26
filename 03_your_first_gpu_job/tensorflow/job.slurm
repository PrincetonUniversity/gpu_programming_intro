#!/bin/bash
#SBATCH --job-name=svd-tf        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2023.9
conda activate /scratch/network/jdh4/.gpu_workshop/envs/tf2-gpu

python svd.py
