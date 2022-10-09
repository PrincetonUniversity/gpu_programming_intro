#include <stdio.h>

__global__ void printLoopIndex() {
    printf("%d\n", threadIdx.x);
}

int main() {
  printLoopIndex<<<1, 100>>>();
  cudaDeviceSynchronize();
}
