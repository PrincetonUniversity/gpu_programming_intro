#!/bin/bash
#SBATCH --job-name=cupy-job      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3
conda activate /scratch/network/$USER/py-gpu

python svd_cupy.py
