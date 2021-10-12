# What is a GPU?

A GPU, or Graphics Processing Unit, is an electronic device originally designed for manipulating the images that appear on a computer monitor. However, beginning around 2007, GPUs have become widely used for accelerating computation in various fields including image processing and machine learning.

Relative to the CPU, GPUs have a far greater number of processing cores but with slower clock speeds. Within a block of threads called a warp, each thread carries out the same operation on a different piece of data. This is the SIMT paradigm (single instruction, multiple threads). GPUs tend to have much less memory than what is available to a CPU. For instance, the P100 GPUs on TigerGPU have only 16 GB compared to 256 GB available to the CPU cores. This is an important consideration in designing algorithms and pipelines.

Many of the fastest supercomputers in the world use GPUs (see [Top 500](https://www.top500.org/lists/top500/2021/06/)).

NVIDIA has been the leading player in GPUs for HPC. However, the GPU market landscape changed in May 2019 when the US DoE announced that Frontier, what is expected to be the first exascale supercomputer in the US, would be based on [AMD GPUs](https://www.hpcwire.com/2019/05/07/cray-amd-exascale-frontier-at-oak-ridge/) and CPUs. Princeton has an [MI100 GPU](https://researchcomputing.princeton.edu/amd-mi100-gpu-testing) which you can use for testing.

![cpu-vs-gpu](http://blog.itvce.com/wp-content/uploads/2016/03/032216_1532_DustFreeNVI2.png)

Like a CPU, a GPU has a hierarchical structure with respect to both the execution units and memory. A warp is a unit of 32 threads. NVIDIA GPUs impose a limit of 1024 threads per block. Some integral number of warps are grouped into a streaming multiprocessor (SM). There are tens of SMs per GPU. Each thread has its own memory. There is limited shared memory between a block of threads. And, finally, there is the global memory which is accessible to each grid or collection of blocks.

![ampere](https://developer-blogs.nvidia.com/wp-content/uploads/2021/guc/raD52-V3yZtQ3WzOE0Cvzvt8icgGHKXPpN2PS_5MMyZLJrVxgMtLN4r2S2kp5jYI9zrA2e0Y8vAfpZia669pbIog2U9ZKdJmQ8oSBjof6gc4IrhmorT2Rr-YopMlOf1aoU3tbn5Q.png)

The figure above is a diagram of a streaming multiprocessor (SM) for the [NVIDIA A100 GPU](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/). The A100 is composed of 108 SMs.

# Overview of using a GPU

This is the essence of how every GPU is used as an accelerator:

+ Copy data from the CPU (host) to the GPU (device)

+ Launch a kernel to carry out computations on the GPU

+ Copy data from the GPU (device) back to the CPU (host)

![gpu-overview](https://blogandcode.files.wordpress.com/2013/12/cudac-1.jpeg?w=597&h=372)

Below is psuedocode for a matrix operation performed on the GPU:

```
data = open("input.dat");     # read the data on the CPU
copyToGPU(data);              # copy the data to the GPU
matrix_inverse(data.gpu);     # perform a matrix operation on the GPU
copyFromGPU(data);            # copy the resulting output back to the CPU
write(data, "output.dat");    # write the output to file on the CPU
```

[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) on Traverse enables fast CPU-to-GPU and GPU-to-GPU data transfers with a peak rate of 75 GB/s per direction. The hardware on the other clusters do not allow for direct GPU-GPU transfers.

# What GPU resources does Princeton have?

See the "Hardware Resources" on the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for a complete list.

## Adroit

There are 2 GPU nodes on Adroit: `adroit-h11g1` and `adroit-h11g4`

<pre>
$ ssh &lt;NetID&gt;@adroit.princeton.edu
$ snodes

HOSTNAMES          STATE    CPUS S:C:T    CPUS(A/I/O/T)   CPU_LOAD MEMORY   GRES                                PARTITION          AVAIL_FEATURES
adroit-01          idle     28   2:14:1   0/28/0/28       0.00     128000   (null)                              class              broadwell
adroit-02          idle     28   2:14:1   0/28/0/28       0.00     128000   (null)                              class              broadwell
adroit-03          idle     28   2:14:1   0/28/0/28       0.00     128000   (null)                              class              broadwell
adroit-04          idle     28   2:14:1   0/28/0/28       0.00     128000   (null)                              class              broadwell
adroit-05          idle     28   2:14:1   0/28/0/28       0.26     128000   (null)                              class              broadwell
adroit-06          idle     28   2:14:1   0/28/0/28       0.00     128000   (null)                              class              broadwell
adroit-07          idle     28   2:14:1   0/28/0/28       0.00     128000   (null)                              class              broadwell
adroit-08          mix      32   2:16:1   19/13/0/32      22.80    384000   (null)                              all*               skylake
adroit-09          mix      32   2:16:1   25/7/0/32       11.09    384000   (null)                              all*               skylake
adroit-10          mix      32   2:16:1   19/13/0/32      12.05    384000   (null)                              all*               skylake
adroit-11          mix      32   2:16:1   2/30/0/32       0.00     384000   (null)                              all*               skylake
adroit-12          mix      32   2:16:1   13/19/0/32      3.02     384000   (null)                              all*               skylake
adroit-13          mix      32   2:16:1   26/6/0/32       25.25    384000   (null)                              all*               skylake
adroit-14          mix      32   2:16:1   13/19/0/32      4.18     384000   (null)                              all*               skylake
adroit-15          mix      32   2:16:1   1/31/0/32       1.02     384000   (null)                              all*               skylake
adroit-16          mix      32   2:16:1   30/2/0/32       44.29    384000   (null)                              all*               skylake
<b>adroit-h11g1       mix      40   2:20:1   1/39/0/40       1.10     770000   gpu:tesla_v100:4(S:0-1)             gpu                v100</b>
<b>adroit-h11g2       idle     48   2:24:1   0/48/0/48       0.00     1000000  gpu:nvidia_a100:4(S:0-1)            gpu                a100</b>
adroit-h11n1       mix      128  2:64:1   18/110/0/128    0.00     256000   (null)                              class              amd,rome
</pre>

### adroit-h11g1

This node has 4 NVIDIA V100 GPUs with 32 GB of memory each. See the specs for the [V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184) or consider buying on [Amazon](https://www.amazon.com/NVIDIA-Tesla-Volta-Accelerator-Graphics/dp/B07JVNHFFX/ref=sr_1_2?keywords=nvidia+v100&qid=1572464893&sr=8-2). Each GPU has 80 streaming multiprocessors (SM) and 64 CUDA cores per SM (and 8 Tensor Cores per SM).

Add this line to your Slurm script to use a V100 GPUs:

```
#SBATCH --gres=gpu:1
```

Here is some information about the V100 GPUs in this node:

```
  CUDADevice with properties:

                      Name: 'Tesla V100-PCIE-32GB'
                     Index: 1
         ComputeCapability: '7.0'
            SupportsDouble: 1
             DriverVersion: 10.1000
            ToolkitVersion: 10
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 3.4058e+10
           AvailableMemory: 3.3552e+10
       MultiprocessorCount: 80
              ClockRateKHz: 1380000
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 0
          CanMapHostMemory: 1
           DeviceSupported: 1
            DeviceSelected: 1
```

Here is infomation about the CPUs on this node:

```
$ ssh <NetID>@adroit.princeton.edu
$ ssh adroit-h11g1
$ lscpu | grep -v Flags

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                32
On-line CPU(s) list:   0-31
Thread(s) per core:    1
Core(s) per socket:    <b>16</b>
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 85
Model name:            Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz
Stepping:              4
CPU MHz:               1526.293
CPU max MHz:           3700.0000
CPU min MHz:           1000.0000
BogoMIPS:              5200.00
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              1024K
L3 cache:              22528K
NUMA node0 CPU(s):     0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
NUMA node1 CPU(s):     1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
```

To see a wealth of information about the GPUs use:

```
$ nvidia-smi -q
```

### adroit-h11g4

`adroit-h11g4` has 2 NVIDIA K40c GPUs with 12 GB of memory per GPU. Each GPU has 15 streaming multiprocessors (SM) and 192 CUDA cores per SM. View the technical specifications for the [K40c](https://www.techpowerup.com/gpu-specs/tesla-k40c.c2505) or buy this GPU on [Amazon](https://www.amazon.com/NVIDIA-Tesla-K40c-computing-processor/dp/B06VSWDH15/ref=sr_1_3?keywords=nvidia+k40c&qid=1572468693&sr=8-3).

Add this line to your Slurm script to use a K40c GPU:

```
#SBATCH --gres=gpu:1
#SBATCH --constraint=k40
```

Here is infomation about the K40c GPUs on this node:


```
  CUDADevice with properties:

                      Name: 'Tesla K40c'
                     Index: 1
         ComputeCapability: '3.5'
            SupportsDouble: 1
             DriverVersion: 10.1000
            ToolkitVersion: 10
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 1.1997e+10
           AvailableMemory: 1.1841e+10
       MultiprocessorCount: 15
              ClockRateKHz: 745000
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 0
          CanMapHostMemory: 1
           DeviceSupported: 1
            DeviceSelected: 1
```

Here is infomation about the CPUs on this node:

```
$ ssh adroit-h11g1
$ lscpu | grep -v Flags

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                16
On-line CPU(s) list:   0-15
Thread(s) per core:    1
Core(s) per socket:    8
Socket(s):             2
NUMA node(s):          2
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 63
Model name:            Intel(R) Xeon(R) CPU E5-2667 v3 @ 3.20GHz
Stepping:              2
CPU MHz:               2738.281
CPU max MHz:           3600.0000
CPU min MHz:           1200.0000
BogoMIPS:              6399.96
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              20480K
NUMA node0 CPU(s):     0-7
NUMA node1 CPU(s):     8-15
```

To see a wealth of information about the GPUs use:

```
$ nvidia-smi -q
```

### Compute Capability and Building Optimized Codes

Some software will only run on a GPU of a given compute capability. To find these values for a given NVIDIA Telsa card see [this page](https://en.wikipedia.org/wiki/Nvidia_Tesla). The compute capability of the V100's in Adroit is 7.0. For various build systems this translates to `sm_70`.

The following is from `$ nvcc --help`:

```
Options for steering GPU code generation.
=========================================

--gpu-architecture <arch>                  (-arch)                         
        Specify the name of the class of NVIDIA 'virtual' GPU architecture for which
        the CUDA input files must be compiled.
        With the exception as described for the shorthand below, the architecture
        specified with this option must be a 'virtual' architecture (such as compute_50).
        Normally, this option alone does not trigger assembly of the generated PTX
        for a 'real' architecture (that is the role of nvcc option '--gpu-code',
        see below); rather, its purpose is to control preprocessing and compilation
        of the input to PTX.
        For convenience, in case of simple nvcc compilations, the following shorthand
        is supported.  If no value for option '--gpu-code' is specified, then the
        value of this option defaults to the value of '--gpu-architecture'.  In this
        situation, as only exception to the description above, the value specified
        for '--gpu-architecture' may be a 'real' architecture (such as a sm_50),
        in which case nvcc uses the specified 'real' architecture and its closest
        'virtual' architecture as effective architecture values.  For example, 'nvcc
        --gpu-architecture=sm_50' is equivalent to 'nvcc --gpu-architecture=compute_50
        --gpu-code=sm_50,compute_50'.
        Allowed values for this option:  'compute_30','compute_32','compute_35',
        'compute_37','compute_50','compute_52','compute_53','compute_60','compute_61',
        'compute_62','compute_70','compute_72','sm_30','sm_32','sm_35','sm_37','sm_50',
        'sm_52','sm_53','sm_60','sm_61','sm_62','sm_70','sm_72'.
```

Hence, a starting point for optimization flags for the A100 GPUs on della-gpu and Adroit:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_80 -o myapp myapp.cu
```

For the V100's on Adroit or Traverse would be:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_70 -o myapp myapp.cu
```

For the P100 GPUs on TigerGPU:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_60 -o myapp myapp.cu
```

## TigerGPU

TigerGPU is composed of 80 Intel Broadwell nodes each with four NVIDIA P100 GPUs. See the P100 [technical specs](https://www.techpowerup.com/gpu-specs/tesla-p100-pcie-16-gb.c2888) or buy on [Amazon](https://www.amazon.com/NVIDIA-Tesla-Passive-Accelerator-900-2H400-0000-000/dp/B0792FXS2S/ref=sr_1_1?keywords=nvidia+p100&qid=1572465106&sr=8-1). Each GPU has 56 streaming multiprocessors (SM) and 64 CUDA FP32 cores per SM.

All the GPUs are the same so to request a GPU add this line to your Slurm script:

```
#SBATCH --gres=gpu:1
```

The following was obtained by running a MATLAB script:

```
  CUDADevice with properties:

                      Name: 'Tesla P100-PCIE-16GB'
                     Index: 1
         ComputeCapability: '6.0'
            SupportsDouble: 1
             DriverVersion: 10.1000
            ToolkitVersion: 9.1000
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 1.7072e+10
           AvailableMemory: 1.6695e+10
       MultiprocessorCount: 56
              ClockRateKHz: 1328500
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 0
          CanMapHostMemory: 1
           DeviceSupported: 1
            DeviceSelected: 1
```

## Traverse

This [new cluster](https://www.princeton.edu/news/2019/10/07/princetons-new-supercomputer-traverse-accelerate-scientific-discovery-fusion) consists of 46 IBM Power9 nodes with 4 NVIDIA V100 GPUs. It is a smaller version of the [Sierra](https://en.wikipedia.org/wiki/Sierra_(supercomputer)) supercomputer. The GPUs on Traverse have 32 GB of memory each and a clock rate of 1.29 GHz. Each GPU has 80 streaming multiprocessors (SM) and 64 CUDA cores per SM (and 8 Tensor Cores per SM).

Additional info:

```
$ ssh traverse-k02g3
$ nvidia-smi -q
```

## Comparison of GPU Resources

|   Cluster  | Number of Nodes | GPUs per Node | NVIDIA GPU Model | GPU Clock (GHz) | Number of FP32 Cores| SM Count | GPU Memory (GB) |
|:----------:|:----------:|:---------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Adroit     |      1           |     4         |  V100    |   1.230        | 5120   | 80  | 32 |
| Adroit     |      1           |     2         |  K40c    |   0.745       | 2880   | 15  | 12 |     
| TigerGPU   |     80           |     4         |  P100    |   1.189        | 3584   | 56  | 16 |
| Traverse   |     46           |     4         |  V100    |   1.290        | 5120   | 80  | 32 | 
| Della      |     20           |     2         |  A100    |   1.410        | 6912   | 108  | 40 | 


SM is streaming multiprocessor. Note that the V100 GPUs have 640 [Tensor Cores](https://devblogs.nvidia.com/cuda-9-features-revealed/) (8 per SM) where half-precision Warp Matrix-Matrix and Accumulate (WMMA) operations can be carried out. That is, each core can perform a 4x4 matrix-matrix multiply and add the result to a third matrix. There are differences between the V100 node on Adroit and the Traverse nodes (see [PCIe versus SXM2](https://www.nextplatform.com/micro-site-content/achieving-maximum-compute-throughput-pcie-vs-sxm2/)).


## GPU Hackathon at Princeton

The next hackathon will take place in June of 2022. This is a great opportunity to get help from experts to port your code to a GPU. Or you can participant as a mentor and help a team rework their code. See the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for details.
