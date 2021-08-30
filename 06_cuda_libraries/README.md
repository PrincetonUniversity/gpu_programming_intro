# GPU-Accelerated Libraries

Let's say you have a CPU code and you are thinking about writing GPU kernels to accelerate the performance of certain parts of the code. Before doing this, you should see if there are GPU libraries that already have implemented the routines that you need. This page presents an overview of the NVIDIA GPU-accelerated libraries.

According to NVIDIA: "NVIDIA GPU-accelerated libraries provide highly-optimized functions that perform 2x-10x faster than CPU-only alternatives. Using drop-in interfaces, you can replace CPU-only libraries such as MKL, IPP and FFTW with GPU-accelerated versions with almost no code changes. The libraries can optimally scale your application across multiple GPUs."

![NVIDIA-GPU-Libraries](https://developer.nvidia.com/sites/default/files/pictures/2017/acceleration.png)

### Selected libraries

+ **cuDNN** - GPU-accelerated library of primitives for deep neural networks
+ **cuBLAS** - GPU-accelerated standard BLAS library
+ **cuSPARSE** - GPU-accelerated BLAS for sparse matrices
+ **cuRAND** - GPU-accelerated random number generation (RNG)
+ **cuSOLVER** - Dense and sparse direct solvers for computer vision, CFD and other applications
+ **cuFFT** - GPU-accelerated library for Fast Fourier Transforms
+ **NPP** - GPU-accelerated image, video, and signal processing functions
+ **NCCL** - Collective Communications Library for scaling apps across multiple GPUs and nodes
+ **nvGRAPH** - GPU-accelerated library for graph analytics

For the complete list see [GPU libraries](https://developer.nvidia.com/gpu-accelerated-libraries) by NVIDIA.

## Where to find the libraries

Run the commands below to examine the libraries:

```
$ module show cudatoolkit/11.0
$ ls -lL /usr/local/cuda-11.0/lib64/lib*.so
```

## Example

Make sure that you are on the head node:

```
$ hostname
adroit4
```

Instead of computing the singular value decomposition (SVD) on the CPU, this example computes it on the GPU. First look over the source code:

```
$ cd gpu_programming_intro/06_cuda_libraries
$ cat gesvdj_example.cpp | less
```

Next, compile and link the code as follows:

```
$ module load cudatoolkit/11.0
$ g++ -o gesvdj_example gesvdj_example.cpp -lcudart -lcusolver
```

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
U(1,1) = 3.0821892063278467E-01
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

Run the following command to obtain a copy of the [NVIDIA CUDA Samples](https://docs.nvidia.com/cuda/cuda-samples/index.html):

```
$ module load cudatoolkit/11.0
$ mkdir ~/nvidia_samples
$ /usr/local/cuda-11.0/bin/cuda-install-samples-11.0.sh ~/nvidia_samples
```

Then browse the directories:

```
$ cd ~/nvidia_samples/NVIDIA_CUDA-11.0_Samples
$ ls -ltrh
total 84K
drwxr-xr-x. 52 jdh4 cses 4.0K Oct 25 16:17 0_Simple
drwxr-xr-x.  8 jdh4 cses  173 Oct 25 16:17 1_Utilities
drwxr-xr-x. 13 jdh4 cses 4.0K Oct 25 16:17 2_Graphics
drwxr-xr-x. 23 jdh4 cses 4.0K Oct 25 16:17 3_Imaging
drwxr-xr-x. 10 jdh4 cses  245 Oct 25 16:17 4_Finance
drwxr-xr-x. 10 jdh4 cses  186 Oct 25 16:17 5_Simulations
drwxr-xr-x. 34 jdh4 cses 4.0K Oct 25 16:17 6_Advanced
drwxr-xr-x. 40 jdh4 cses 4.0K Oct 25 16:17 7_CUDALibraries
drwxr-xr-x.  6 jdh4 cses   95 Oct 25 16:18 common
-rw-r--r--.  1 jdh4 cses  59K Oct 25 16:18 EULA.txt
-rw-r--r--.  1 jdh4 cses 2.6K Oct 25 16:18 Makefile
```

Pick an example and then build and run it. For instance:

```
$ cd 0_Simple/matrixMul
$ make TARGET_ARCH=x86_64 SMS="70" HOST_COMPILER=g++  # use 60 on tigergpu and 80 on della-gpu
$ cp gpu_programming_intro/06_cuda_libraries/job.slurm .
```

Edit the Slurm script by **changing the last line** as follows:

```bash
#!/bin/bash
#SBATCH --job-name=cuda-libs     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module load cudatoolkit/11.0

./matrixMul
```

Submit the job:

```
$ sbatch job.slurm
```

Note that some examples have dependencies that will not be satisfied so they will not build. This can be resolved if it relates to your research work. For instance, to build `5_Simulations/nbody` use:

```
GLPATH=/lib64 make TARGET_ARCH=x86_64 SMS="70" HOST_COMPILER=g++  # use 60 on tigergpu and 80 on della-gpu
```

Note that `nbody` will run successfully on TigerGPU but not on Adroit since the compute nodes do not have `libglut.so`.
