# Built-in Thread and Block Indices

Each thread is given an index within its thread block, starting at 0. Additionally, each block is given an index, starting at 0. Just as threads are grouped into thread blocks, blocks are grouped into a grid, which is the highest entity in the CUDA thread hierarchy.

![intrinic-indices](https://devblogs.nvidia.com/wp-content/uploads/2017/01/cuda_indexing.png)

CUDA kernels have access to special variables identifying both the index of the thread (within the block) that is executing the kernel, and, the index of the block (within the grid) that the thread is within. These variables are `threadIdx.x` and `blockIdx.x` respectively. Below is an example use of `threadIdx.x`:

```C
__global__ void GPUFunction() {
  printf("My thread index is: %g\n", threadIdx.x);
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
$ cd 
$ module load cudatoolkit
$ nvcc -o for_loop for_loop.c
$ sbatch job.slurm
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

Click [here](hint.md) to see some hints.

One possible solution is [here](solution.cu).
