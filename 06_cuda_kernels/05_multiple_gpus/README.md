# Multiple GPUs

The code in the this directory illustrates the use of multiple GPUs. To compile and execute the example, run the following commands:

```
$ module load cudatoolkit/11.7
$ nvcc -O3 -arch=sm_80 -o multi_gpu multi_gpu.cu
$ sbatch job.slurm
```

On Traverse and the Adroit V100 nodes, replace `sm_80` with `sm_70`.

See also `Samples/0_Introduction/simpleMultiGPU` in the NVIDIA samples which are discussed in `05_cuda_libraries`.
