# Multiple GPUs

The code in the this directory illustrates the use of multiple GPUs. To compile and execute the example, run the following commands:

```
$ module load cudatoolkit
$ nvcc -O3 -arch=sm_y0 -o vector_add_gpu vector_add_gpu.cu
$ sbatch job.slurm
```
