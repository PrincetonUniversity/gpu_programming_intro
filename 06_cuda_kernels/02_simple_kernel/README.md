# Launching Parallel Kernels

The execution configuration allows programmers to specify details about launching the kernel to run in parallel on multiple GPU threads. More precisely, the execution configuration allows programmers to specifiy how many groups of threads (called thread blocks) and how many threads they would like each thread block to contain. The syntax for this is:

```
<<<NUMBER_OF_BLOCKS, NUMBER_OF_THREADS_PER_BLOCK>>>
```

The kernel code is executed by every thread in every thread block configured when the kernel is launched. The image below corresponds to `<<<1, 5>>>`:

![thread-block](https://miro.medium.com/max/1118/1*e_FAITzOXSearSZYNWnmKQ.png)


## CPU Code

```c
#include <stdio.h>

void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{
  firstParallel();
}
```

## Exercise: GPU implementation

```
# rewrite the CPU code above so that it runs on a GPU using multiple threads
# save your file as first_parallel.cu (a starting file by this name is given -- see below)
```

The objective is to write a GPU code with one kernel launch that produces the following 6 lines of output:

```
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
```

To get started:

```
$ module load cudatoolkit/12.8
$ cd gpu_programming_intro/06_cuda_kernels/02_simple_kernel
# edit first_parallel.cu   (use a text editor of your choice)
$ nvcc -o first_parallel first_parallel.cu
$ sbatch job.slurm
```

There are multiple possible solutions.
