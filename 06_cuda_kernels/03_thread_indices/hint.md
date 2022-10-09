## Hints

To understand how to do this exercise, take a look at the code below which uses `threadIdx.x`:

```C
#include <stdio.h>

__global__ void GPUFunction() {
  printf("My thread index is: %g\n", threadIdx.x);
}

int main() {
  GPUFunction<<<1, 1>>>();
  cudaDeviceSynchronize();
}
```

The output of the code above is

```
My thread index is: 0
```

We need to replace the i variable in the CPU code. In a CUDA kernel, each thread has an index
associated with it called `threadIdx.x`. So use that as the substitution for i.

Next, to generate 100 threads, try a kernel launch like this: `<<<1, 100>>>`

The above will give you 1 block composed of 100 threads.

Be sure to add `__global__` to your GPU function and don't forget to call `cudaDeviceSynchronize()`.
