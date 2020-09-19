# Multiple GPUs

The code in the this directory illustrates the use of multiple GPUs. To compile and execute the example, run the following commands:

```
$ module load cudatoolkit
$ nvcc -O3 -arch=sm_70 -o multi_gpu multi_gpu.cu
$ sbatch job.slurm
```

On TigerGPU, replace `sm_70` with `sm_60`.

See also `0_Simple/simpleMultiGPU` in the NVIDIA samples which are discussed in `06_cuda_libraries`.
