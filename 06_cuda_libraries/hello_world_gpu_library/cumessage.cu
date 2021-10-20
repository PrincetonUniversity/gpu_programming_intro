#include <stdio.h>
#include "cumessage.h"

__global__ void GPUFunction_kernel() {
  printf("Hello world from the GPU.\n");
}

void GPUFunction() {
  GPUFunction_kernel<<<1,1>>>();
  
  // kernel execution is asynchronous so sync on its completion
  cudaDeviceSynchronize();
}
