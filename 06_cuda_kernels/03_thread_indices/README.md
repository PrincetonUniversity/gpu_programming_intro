# Built-in Thread and Block Indices

Each thread is given an index within its thread block, starting at 0. Additionally, each block is given an index, starting at 0. Threads are grouped into thread blocks, blocks are grouped into grids, and grids can be grouped into a cluster, which is the highest entity in the CUDA hierarchy.

![intrinic-indices](https://devblogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)

CUDA kernels have access to special variables identifying both the index of the thread (within the block) that is executing the kernel, and, the index of the block (within the grid) that the thread is within. These variables are `threadIdx.x` and `blockIdx.x` respectively. Below is an example use of `threadIdx.x`:

```C
__global__ void GPUFunction() {
  printf("My thread index is: %d\n", threadIdx.x);
}
```

## CPU implentation of a for loop

```C
#include <stdio.h>

void printLoopIndex() {
  int N = 100;
  for (int i = 0; i < N; ++i)
    printf("%d\n", i);
}

int main() {
  // function to run on the cpu
  printLoopIndex();
}
```

Run the CPU code above by following these commands:

```bash
$ cd gpu_programming_intro/06_cuda_kernels/03_thread_indices
$ nvcc -o for_loop for_loop.c
$ ./for_loop
```

The output of the above is

```
0
1
2
...
97
98
99
```

## Exercise: GPU implementation

In the CPU code above, the loop is carried out in serial. That is, loop iterations takes place one at a time. Can you write a GPU code that produces the same output as that above but does so in parallel using a CUDA kernel?

```
// write a GPU kernel to produce the output above
```

To get started:

```bash
$ module load cudatoolkit/12.8
# edit for_loop.cu
$ nvcc -o for_loop for_loop.cu
$ sbatch job.slurm
```

Click [here](hint.md) to see some hints.

One possible solution is [here](solution.cu) (try for yourself first).

Are you seeing any behavior which is a multiple of 32 in this exercise? For NVIDIA, the threads within a thread block are organized into "warps". A "warp" is composed of 32 threads. [Read more](http://15418.courses.cs.cmu.edu/spring2013/article/15) about how `printf` works in CUDA.
