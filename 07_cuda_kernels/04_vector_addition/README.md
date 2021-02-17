# Elementwise Vector Addition

## A Word on Allocating Memory

Here is an example on the CPU where 10 integers are dynamically allocated and the last line frees the memory:

```C
int N = 10;
size_t size = N * sizeof(int);

int *a;
a = (int*)malloc(size);
free(a);
```

On the GPU:

```C
int N = 10;
size_t size = N * sizeof(int);

int *d_a;
cudaMalloc(&d_a, size);
cudaFree(d_a);
```
Note that we write `d_a` for the GPU case instead of `a` to remind ourselves that we are allocating memory on the "device" or GPU. Sometimes developers with prefix CPU variables with 'h' to denote "host".

![add-arrays](https://www3.ntu.edu.sg/home/ehchua/programming/cpp/images/Array.png)

## CPU

The following code adds two vectors together on a CPU:

```C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
 
void vecAdd(double *a, double *b, double *c, int n)
{
    int i;
    for(i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 2000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }

    // add the two vectors
    vecAdd(h_a, h_b, h_c, n);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

Take a look at `vector_add_cpu.c`. You will see that it allocates three arrays of size `n` and then fills `a` and `b` with values. The `vecAdd` function is then called to perform the elementwise addition of the two arrays producing a third array `c`:

```C
void vecAdd(double *a, double *b, double *c, int n) {
    int i;
    for(i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```


The output reports the time taken to perform the addition ignoring the memory allocation and initialization. Build and run the code:

```
$ cd gpu_programming_intro/07_cuda_kernels/04_vector_addition
$ gcc -O3 -march=native -o vector_add_cpu vector_add_cpu.c -lm
$ ./vector_add_cpu
```

## GPU

The following code adds two vectors together on a GPU:

```C
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
 
// each thread is responsible for one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
 
    // Make sure we do not go out of bounds
    int i;
    for (i = id; i < n; i += stride)
      c[i] = a[i] + b[i];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int n = 2000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
 
    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    int blockSize, gridSize;
 
    // Number of threads in each thread block
    blockSize = 1024;
 
    // Number of thread blocks in grid
    gridSize = (int)ceil((double)n/blockSize);
    if (gridSize > 65535) gridSize = 32000;
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceSynchronize();
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

The `vecAdd` function has been replaced with a CUDA kernel:

```C
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
 
    // Make sure we do not go out of bounds
    int i;
    for (i = id; i < n; i += stride)
      c[i] = a[i] + b[i];
}
```

The kernel uses special variables which are CUDA extensions to allow threads to distinguish themselves and operate on different data. Specifically, `blockIdx.x` is the block index within a grid, `blockDim.x` is the number of threads per block and `threadIdx.x` is the thread index within a block. Let's build and run the code. The `nvcc` compiler will compile the kernel function while `gcc` will be used in the background to compile the CPU code.

```
$ module load cudatoolkit/10.2
$ nvcc -O3 -arch=sm_60 -o vector_add_gpu vector_add_gpu.cu
$ sbatch job.slurm
```

The output of the code will be something like:
```
Allocating CPU memory and populating arrays of length 2000 ... done.
GridSize 2 and total_threads 2048
Performing vector addition (timer started) ... done in 0.19 s.
```

Note that the reported time include all operations beyond those needed to carry out the operation on the GPU. This includes the time required to allocate and deallocate memory on the GPU and the time required to move the data to and from the GPU.

To use a GPU effectively the problem you are solving must have a vast amount of data parallelism. In the example here one can assign a different thread to each of the individual elements. For problems involving recursion or sorting or small amounts of data, it becomes difficult to take advantage of a GPU.

## Advanced Examples

For more advanced examples return to the NVIDIA CUDA samples at the bottom of [this page](https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/06_cuda_libraries).
