#include <stdio.h>

void CPUFunction() {
  printf("Hello world from the CPU.\n");
}

__global__ void GPUFunction(int myid) {
  printf("Hello world from GPU %d.\n", myid);
}

int main() {

  // function to run on the cpu
  CPUFunction();

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  for (device=0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
  }

  // run on gpu 0
  int device_id = 0;
  cudaSetDevice(device_id);
  GPUFunction<<<1, 1>>>(device_id);
 
  // run on gpu 1
  device_id = 1;
  cudaSetDevice(device_id);
  GPUFunction<<<1, 1>>>(device_id);

  // kernel execution is asynchronous so sync on its completion
  cudaDeviceSynchronize();
}
