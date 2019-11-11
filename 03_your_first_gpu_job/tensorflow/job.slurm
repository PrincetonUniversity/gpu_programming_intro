#!/bin/bash
#SBATCH --job-name=svd-tf        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1G                 # total memory (RAM) per node
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --reservation=introgpu   # REMOVE THIS LINE AFTER THE WORKSHOP

module load anaconda3
conda activate /scratch/network/$USER/tf-gpu

srun python svd_tensorflow.py
