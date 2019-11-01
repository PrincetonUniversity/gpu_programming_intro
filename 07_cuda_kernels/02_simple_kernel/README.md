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
# rewrite the CPU code above so that it runs on a GPU
# save your file as first_parallel.cu (a starting file by this name is given -- see below)
```

To be clear, you are trying to write a GPU code with one kernel launch that produces something like the following:

```
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
This should be running in parallel.
```

Run your GPU code with different values of `NUMBER_OF_BLOCKS` and `NUMBER_OF_THREADS_PER_BLOCK` to see how the execution configuration works.

```
$ cd gpu_programming_intro/07_cuda_kernels/02_simple_kernel
# edit first_parallel.cu   (use a text editor of your choice)
$ nvcc -o first_parallel first_parallel.cu
$ sbatch job.slurm
```

One possible solution to this exercise is [here](solution.cu).
