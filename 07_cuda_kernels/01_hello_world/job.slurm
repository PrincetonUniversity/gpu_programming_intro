#!/bin/bash
#SBATCH --job-name=hw-gpu        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=1G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:tesla_v100:1  # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

./hello_world_gpu
