#!/bin/bash
#SBATCH --job-name=gpu-lib       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=fallgpu    # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load cudatoolkit/11.7
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH

./myapp
