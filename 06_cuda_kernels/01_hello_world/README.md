# Hello World

On this page we consider the simplest CPU C code and the simplest CUDA C GPU code.

## CPU

A simple CPU-only code:

```C
#include <stdio.h>

void CPUFunction() {
  printf("Hello world from the CPU.\n");
}

int main() {
  // function to run on the cpu
  CPUFunction();
}
```

This can be compiled and run with:

```
$ cd gpu_programming_intro/06_cuda_kernels/01_hello_world
$ gcc -o hello_world hello_world.c
$ ./hello_world
```

The output is

```
Hello world from the CPU.
```

## GPU

Below is a simple GPU code that calls a CPU function followed by a GPU function:

```C
#include <stdio.h>

void CPUFunction() {
  printf("Hello world from the CPU.\n");
}

__global__ void GPUFunction() {
  printf("Hello world from the GPU.\n");
}

int main() {
  // function to run on the cpu
  CPUFunction();

  // function to run on the gpu
  GPUFunction<<<1, 1>>>();
  
  // kernel execution is asynchronous so sync on its completion
  cudaDeviceSynchronize();
}
```

The GPU code above can be compiled and executed with:

```
$ module load cudatoolkit/12.2
$ nvcc -o hello_world_gpu hello_world_gpu.cu
$ sbatch job.slurm
```

The output should be:

```
Hello world from the CPU.
Hello world from the GPU.
```

`nvcc` is the NVIDIA CUDA Compiler. It compiles the GPU code itself and uses GNU `gcc` to compile the CPU code. CUDA provides extensions for many common programming languages (e.g., C/C++/Fortran). These language extensions allow developers to write GPU functions.

From this simple example we learn that GPU functions are declared with `__global__`, which is a CUDA C/C++ keyword. The triple angle brackets or so-called "triple chevron" is used to specify the execution configuration of the kernel launch which is a call from host code to device code.

Here is the general form for the execution configuration: `<<<NumBlocks, NumThreadsPerBlock>>>`. In the example above we used 1 block and 1 thread per block. At a high level, the execution configuration allows programmers to specify the thread hierarchy for a kernel launch, which defines the number of thread groupings (called blocks), as well as how many threads to execute in each block.

Notice the return type of `void` for GPUFunction. It is required that GPU functions defined with the `__global__` keyword return type `void`.

### Exercises

1. What happens if you remove `__global__`?

2. Can you rewrite the code so that the output is:

```
Hello world from the CPU.
Hello world from the GPU.
Hello world from the CPU.
```

3. What happens if you comment out the `cudaDeviceSynchronize()` line by preceding it with `//`?
