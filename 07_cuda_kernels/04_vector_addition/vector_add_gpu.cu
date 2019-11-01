/* GPU Version */

// original file is https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"
 
// CUDA kernel. Each thread takes care of one element of c
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
    fprintf(stderr, "Allocating CPU memory and populating arrays of length %d ...", n);
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
    fprintf(stderr, " done.\n");

    fprintf(stderr, "Performing vector addition (timer started) ...");
    StartTimer();

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
    printf("GridSize %d and total_threads %d\n", gridSize, gridSize * blockSize); 
    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
 
    // Copy array back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost );
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaDeviceSynchronize();
 
    double runtime = GetTimer();
    fprintf(stderr, " done in %.2f s.\n", runtime / 1000);
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    double tol = 1e-6;
    //printf("\nout is %f\n", sum/n);
    if (fabs(sum/n - 1.0) > tol) printf("Warning: potential numerical problems.\n");
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
