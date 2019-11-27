# Advanced and Other

## CUDA-Aware MPI

On TigerGPU you will see MPI modules that have been built against CUDA. These modules enable [CUDA-aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) where
memory on a GPU can be sent to another GPU without concerning a CPU. According to NVIDIA:

> Regular MPI implementations pass pointers to host memory, staging GPU buffers through host memory using cudaMemcopy.

> With [CUDA-aware MPI](https://developer.nvidia.com/mpi-solutions-gpus), the MPI library can send and receive GPU buffers directly, without having to first stage them in host memory. Implementation of CUDA-aware MPI was simplified by Unified Virtual Addressing (UVA) in CUDA 4.0 – which enables a single address space for all CPU and GPU memory. CUDA-aware implementations of MPI have several advantages.

See the CUDA-aware MPI modules on TigerGPU:

```
$ ssh <NetID>@tigergpu.princeton.edu
$ module avail openmpi/cuda

------------------------------------ /usr/local/share/Modules/modulefiles ------------------------------------
openmpi/cuda-8.0/gcc/2.1.0/64        openmpi/cuda-8.0/intel-17.0/2.1.0/64 openmpi/cuda-9.0/gcc/3.0.0/64
openmpi/cuda-8.0/gcc/3.0.0/64        openmpi/cuda-8.0/intel-17.0/3.0.0/64 openmpi/cuda-9.0/intel-17.0/3.0.0/64
```

## GPU Direct

[GPU Direct](https://developer.nvidia.com/gpudirect) is a solution to the problem of data-starved GPUs.

![gpu-direct](https://developer.nvidia.com/sites/default/files/akamai/GPUDirect/cuda-gpu-direct-blog-refresh_diagram_1.png)

> Using GPUDirect™, multiple GPUs, network adapters, solid-state drives (SSDs) and now NVMe drives can directly read and write CUDA host and device memory, eliminating unnecessary memory copies, dramatically lowering CPU overhead, and reducing latency, resulting in significant performance improvements in data transfer times for applications running on NVIDIA Tesla™ and Quadro™ products

## GPU Sharing

Many GPU applications only use the GPU for a fraction of the time. For many years, a goal of GPU vendors has been to allow for GPU sharing between applications. Slurm is capable of supporting this through the `--gpu-mps` option.

## OpenCL/SYCL

NVIDIA is a private company. If you chose to organize your GPU software and hardware around CUDA then you are locked in to a single vendor. A non-proprietary alternative to NVIDIA CUDA is the Open Computing Language or [OpenCL](https://www.khronos.org/opencl/).

## OpenMP 4.5+

Recent implementations of [OpenMP](https://www.openmp.org/) support GPU programming. However, they are not mature and should not be favored.

## CUDA Kernels versus OpenACC on the Long Term

CUDA kernels are written at a low level. OpenACC is a high-level programmaing model. Because GPU hardware is changing rapidly, some argue that writing GPU codes with OpenACC is a better choice because there is much less work do to when new hardware comes out.

[See the materials](http://w3.pppl.gov/~ethier/PICSCIE/Intro_to_OpenACC_Nov_2019.pdf) for an OpenACC workshop by Stephane Ethier.

## Using the Intel Compiler for Host Code

Note the use of `auto` in the code below:

```c++
#include <stdio.h>

__global__ void simpleKernel()
{
  auto i = blockDim.x * blockIdx.x + threadIdx.x;
  printf("Index: %d\n", i);
}

int main()
{
  simpleKernel<<<2, 3>>>();
  cudaDeviceSynchronize();
}
```

The C++11 language standard introduced the `auto` keyword. To compile the code with the Intel compiler for TigerGPU:

```
$ module load intel
$ nvcc -ccbin=icpc -std=c++11 -arch=sm_60 -o simple simple.cu
```

Note that you may need to also load the `rh` module to get the compilation to work. The `rh` module make a newer GCC available. The Intel compiler depends on GCC.
