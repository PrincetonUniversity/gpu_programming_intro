# GPU-Accelerated Libraries

Let's say you have a CPU code and you are thinking about writing GPU kernels to accelerate the performance of the slow parts of the code. Before doing this, you should see if there are GPU libraries that already have implemented the routines that you need. This page presents an overview of the NVIDIA GPU-accelerated libraries.

According to NVIDIA: "NVIDIA GPU-accelerated libraries provide highly-optimized functions that perform 2x-10x faster than CPU-only alternatives. Using drop-in interfaces, you can replace CPU-only libraries such as MKL, IPP and FFTW with GPU-accelerated versions with almost no code changes. The libraries can optimally scale your application across multiple GPUs."

![NVIDIA-GPU-Libraries](https://tigress-web.princeton.edu/~jdh4/nv_libraries.jpeg)

### Selected libraries

+ **cuDNN** - GPU-accelerated library of primitives for deep neural networks
+ **cuBLAS** - GPU-accelerated standard BLAS library
+ **cuSPARSE** - GPU-accelerated BLAS for sparse matrices
+ **cuRAND** - GPU-accelerated random number generation (RNG)
+ **cuSOLVER** - Dense and sparse direct solvers for computer vision, CFD and other applications
+ **cuTENSOR** - GPU-accelerated tensor linear algebra library
+ **cuFFT** - GPU-accelerated library for Fast Fourier Transforms
+ **NPP** - GPU-accelerated image, video, and signal processing functions
+ **NCCL** - Collective Communications Library for scaling apps across multiple GPUs and nodes
+ **nvGRAPH** - GPU-accelerated library for graph analytics

For the complete list see [GPU libraries](https://developer.nvidia.com/gpu-accelerated-libraries) by NVIDIA.

## Where to find the libraries

Run the commands below to examine the libraries:

```
$ module show cudatoolkit/12.2
$ ls -lL /usr/local/cuda-12.2/lib64/lib*.so
```

## Example

Make sure that you are on the `adroit5` login node :

```
$ hostname
adroit5
```

Instead of computing the singular value decomposition (SVD) on the CPU, this example computes it on the GPU using `libcusolver`. First look over the source code:

```
$ cd gpu_programming_intro/05_cuda_libraries
$ cat gesvdj_example.cpp | less  # q to quit
```

The header file `cusolverDn.h` included by `gesvdj_example.cpp` contains the line `cuSolverDN : Dense Linear Algebra Library` providing information about its purpose. See the [cuSOLVER API](https://docs.nvidia.com/cuda/cusolver/index.html) for more.


Next, compile and link the code as follows:

```
$ module load cudatoolkit/12.2
$ g++ -o gesvdj_example gesvdj_example.cpp -lcudart -lcusolver
```

Run `ldd gesvdj_example` to check the linking against cuSOLVER (i.e., `libcusolver.so`).

Submit the job to the scheduler with:

```
$ sbatch job.slurm
```

The ouput should appears as:

```
$ cat slurm-*.out

example of gesvdj
tol = 1.000000E-07, default value is machine zero
max. sweeps = 15, default value is 100
econ = 0
A = (matlab base-1)
A(1,1) = 1.0000000000000000E+00
A(1,2) = 2.0000000000000000E+00
A(2,1) = 4.0000000000000000E+00
A(2,2) = 5.0000000000000000E+00
A(3,1) = 2.0000000000000000E+00
A(3,2) = 1.0000000000000000E+00
=====
gesvdj converges
S = singular values (matlab base-1)
S(1,1) = 7.0652834970827287E+00
S(2,1) = 1.0400812977120775E+00
=====
U = left singular vectors (matlab base-1)
U(1,1) = 3.0821892063278472E-01
U(1,2) = -4.8819507401989848E-01
U(1,3) = 8.1649658092772659E-01
U(2,1) = 9.0613333377729299E-01
U(2,2) = -1.1070553170904460E-01
U(2,3) = -4.0824829046386302E-01
U(3,1) = 2.8969549251172333E-01
U(3,2) = 8.6568461633075366E-01
U(3,3) = 4.0824829046386224E-01
=====
V = right singular vectors (matlab base-1)
V(1,1) = 6.3863583713639760E-01
V(1,2) = 7.6950910814953477E-01
V(2,1) = 7.6950910814953477E-01
V(2,2) = -6.3863583713639760E-01
=====
|S - S_exact|_sup = 4.440892E-16
residual |A - U*S*V**H|_F = 3.511066E-16
number of executed sweeps = 1
```

## NVIDIA CUDA Samples

Run the following command to obtain a copy of the [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples):

```
$ cd gpu_programming_intro
$ git clone https://github.com/NVIDIA/cuda-samples.git
$ cd cuda-samples/Samples
```

Then browse the directories:

```
$ ls -ltrh
total 20K
drwxr-xr-x. 55 jdh4 cses 4.0K Oct  9 18:23 0_Introduction
drwxr-xr-x.  6 jdh4 cses  130 Oct  9 18:23 1_Utilities
drwxr-xr-x. 36 jdh4 cses 4.0K Oct  9 18:23 2_Concepts_and_Techniques
drwxr-xr-x. 25 jdh4 cses 4.0K Oct  9 18:23 3_CUDA_Features
drwxr-xr-x. 40 jdh4 cses 4.0K Oct  9 18:23 4_CUDA_Libraries
drwxr-xr-x. 52 jdh4 cses 4.0K Oct  9 18:23 5_Domain_Specific
drwxr-xr-x.  5 jdh4 cses  105 Oct  9 18:23 6_Performance
```

Pick an example and then build and run it. For instance:

```
$ module load cudatoolkit/12.2
$ cd 0_Introduction/matrixMul
$ make TARGET_ARCH=x86_64 SMS="80" HOST_COMPILER=g++  # use 70 on traverse and adroit v100 node
```

This will produce `matrixMul`. If you run the `ldd` command on `matrixMul` you will see that it does not link against `cublas.so`. Instead it uses a naive implementation of the routine which is surely not as efficient as the library implementation.

```
$ cp <PATH/TO>/gpu_programming_intro/05_cuda_libraries/matrixMul/job.slurm .
```

Submit the job:

```
$ sbatch job.slurm
```

See `4_CUDA_Libraries` for more examples. For instance, take a look at `4_CUDA_Libraries/matrixMulCUBLAS`. Does the resulting executable link against `libcublas.so`?

```
$ cd ../../4_CUDA_Libraries/matrixMulCUBLAS
$ make TARGET_ARCH=x86_64 SMS="80" HOST_COMPILER=g++
$ ldd matrixMulCUBLAS
```

Similarly, does the code in `4_CUDA_Libraries/simpleCUFFT_MGPU` link against `libcufft.so`?

To run code that uses the Tensor Cores see examples such as `3_CUDA_Features/bf16TensorCoreGemm`. That example uses the bfloat16 floating-point format.

Note that some examples have dependencies that will not be satisfied so they will not build. This can be resolved if it relates to your research work. For instance, to build `5_Domain_Specific/nbody` use:

```
GLPATH=/lib64 make TARGET_ARCH=x86_64 SMS="80" HOST_COMPILER=g++  # use 70 on traverse and adroit v100 node
```

Note that `nbody` will not run successfully on adroit since the GPU nodes do not have `libglut.so`. The library could be added if needed. One can compile and run this code on adroit-vis using `TARGET_ARCH=x86_64 SMS="80"`.
