# What is a GPU?

A GPU, or Graphics Processing Unit, is an electronic device originally designed for manipulating the images that appear on a computer monitor. However, beginning around 2007, GPUs have become widely used for accelerating computation in various fields including image processing and machine learning.

Relative to the CPU, GPUs have a far greater number of processing cores but with slower clock speeds. Within a block of threads called a warp (NVIDIA), each thread carries out the same operation on a different piece of data. This is the SIMT paradigm (single instruction, multiple threads). GPUs tend to have much less memory than what is available on a CPU. For instance, the A100 GPUs on Della have 80 GB compared to 1000 GB available to the CPU cores. This is an important consideration when designing algorithms and running jobs. Furthermore, GPUs are intended for highly parallel algorithms. The CPU can often out-perform a GPU on algorithms that are not highly parallelizable such as those that rely on data caching and flow control (e.g., "if" statements).

Many of the fastest supercomputers in the world use GPUs (see [Top 500](https://www.top500.org/lists/top500/2023/06/)). How many of the top 10 supercomputers use GPUs?

NVIDIA has been the leading player in GPUs for HPC. However, the GPU market landscape changed in May 2019 when the US DoE announced that Frontier, the first exascale supercomputer in the US, would be based on [AMD GPUs](https://www.hpcwire.com/2019/05/07/cray-amd-exascale-frontier-at-oak-ridge/) and CPUs. Princeton has a two [MI210 GPUs](https://researchcomputing.princeton.edu/amd-mi100-gpu-testing) which you can use for testing. Intel will soon be a new player when the [Aurora supercomputer](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) is completed.

All laptops have a GPU for graphics. It is becoming standard for a laptop to have a second GPU dedicated for compute (see the latest [MacBook Pro](https://www.apple.com/macbook-pro/)).

![cpu-vs-gpu](http://blog.itvce.com/wp-content/uploads/2016/03/032216_1532_DustFreeNVI2.png)

The image below emphasizes the cache sizes and flow control:

![cache_flow_control](https://tigress-web.princeton.edu/~jdh4/gpu-devotes-more-transistors-to-data-processing.png)

Like a CPU, a GPU has a hierarchical structure with respect to both the execution units and memory. A warp is a unit of 32 threads. NVIDIA GPUs impose a limit of 1024 threads per block. Some integral number of warps are grouped into a streaming multiprocessor (SM). There are tens of SMs per GPU. Each thread has its own memory. There is limited shared memory between a block of threads. And, finally, there is the global memory which is accessible to each grid or collection of blocks.

![ampere](https://developer-blogs.nvidia.com/wp-content/uploads/2021/guc/raD52-V3yZtQ3WzOE0Cvzvt8icgGHKXPpN2PS_5MMyZLJrVxgMtLN4r2S2kp5jYI9zrA2e0Y8vAfpZia669pbIog2U9ZKdJmQ8oSBjof6gc4IrhmorT2Rr-YopMlOf1aoU3tbn5Q.png)

The figure above is a diagram of a streaming multiprocessor (SM) for the [NVIDIA A100 GPU](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/). The A100 is composed of 108 SMs.

# Overview of using a GPU

This is the essence of how every GPU is used as an accelerator for compute:

+ Copy data from the CPU (host) to the GPU (device)

+ Launch a kernel to carry out computations on the GPU

+ Copy data from the GPU (device) back to the CPU (host)

![gpu-overview](https://tigress-web.princeton.edu/~jdh4/gpu_as_accelerator_to_cpu_diagram.png)

The diagram above and the accompanying pseudocode present a simplified view of how GPUs are used in scientific computing. To fully understand how things work you will need to learn more about memory cache, interconnects, CUDA streams and much more.

[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) on Traverse enables fast CPU-to-GPU and GPU-to-GPU data transfers with a peak rate of 75 GB/s per direction. Della has this fast GPU-GPU interconnect on each pair of GPUs on 70 of the 90 GPU nodes.

Given the significant performance penalty for moving data between the CPU and GPU, it is natural to work toward "unifying" the CPU and GPU. For instance, read about the [NVIDIA Grace Superchip](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/).

# What GPU resources does Princeton have?

See the "Hardware Resources" on the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for a complete list.

## Adroit

There are 3 GPU nodes on Adroit: `adroit-h11g1`, `adroit-h11g2` and `adroit-h11g3`

<pre>
$ ssh &lt;NetID&gt;@adroit.princeton.edu
$ shownodes
NODELIST      PART   STATE        FREE/TOTAL CPUs  CPU_LOAD  FREE/TOTAL MEMORY  FREE/TOTAL GPUs          FEATURES
adroit-08     class  idle                   32/32      0.01    376911/384000Mb                      skylake,intel
adroit-09     class  idle                   32/32      0.01    382495/384000Mb                      skylake,intel
adroit-10     class  idle                   32/32      0.00    381901/384000Mb                      skylake,intel
adroit-11     class  mixed                   8/32      0.00    357886/384000Mb                      skylake,intel
adroit-12     class  idle                   32/32      0.00    382242/384000Mb                      skylake,intel
adroit-13     all    allocated               0/32     32.33    279722/384000Mb                      skylake,intel
adroit-14     all    mixed                   8/32     17.45    217066/384000Mb                      skylake,intel
adroit-15     all    allocated               0/32     32.05    211596/384000Mb                      skylake,intel
adroit-16     class  mixed                   6/32     14.70    349978/384000Mb                      skylake,intel
adroit-h11g1  gpu    idle                   48/48      0.00   902550/1000000Mb  4/4 nvidia_a100  a100,intel,gpu80
adroit-h11g2  gpu    mixed                  42/48      2.05   832854/1000000Mb  1/4 nvidia_a100        a100,intel
adroit-h11g3  gpu    mixed                  48/56      7.99    607647/760000Mb   3/4 tesla_v100        v100,intel
adroit-h11n1  class  idle                 128/128      0.00    250889/256000Mb                           amd,rome
adroit-h11n2  all    mixed                   4/64     40.74    244458/512000Mb                          intel,ice
adroit-h11n3  all    mixed                   2/64     41.16    156131/512000Mb                          intel,ice
adroit-h11n4  all    mixed                   8/64     56.03    149451/512000Mb                          intel,ice
adroit-h11n5  all    mixed                   8/64     51.12    283062/512000Mb                          intel,ice
adroit-h11n6  all    mixed                   7/64     48.30    158202/512000Mb                          intel,ice
</pre>

To only see the GPU nodes:

<pre>
$ shownodes -p gpu
NODELIST      STATE    FREE/TOTAL CPUs  CPU_LOAD  FREE/TOTAL MEMORY  FREE/TOTAL GPUs          FEATURES
adroit-h11g1  idle               48/48      0.00   902550/1000000Mb  4/4 nvidia_a100  a100,intel,gpu80
adroit-h11g2  mixed              42/48      2.05   832854/1000000Mb  1/4 nvidia_a100        a100,intel
adroit-h11g3  mixed              48/56      7.99    607647/760000Mb   3/4 tesla_v100        v100,intel
</pre>
  
### adroit-h11g1

This node has 4 NVIDIA A100 GPUs with 80 GB of memory each. Each A100 GPU has 108 streaming multiprocessors (SM) and 64 CUDA cores per SM (and 8 Tensor Cores per SM).

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

<pre>
$ ssh &lt;NetID&gt;@adroit.princeton.edu
$ salloc --nodes=1 --ntasks=1 --mem=4G --time=00:05:00 --gres=gpu:1 --constraint=gpu80
$ lscpu | grep -v Flags
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              48
On-line CPU(s) list: 0-47
Thread(s) per core:  1
Core(s) per socket:  24
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6
Model:               143
Model name:          Intel(R) Xeon(R) Gold 6442Y
Stepping:            8
CPU MHz:             3707.218
CPU max MHz:         4000.0000
CPU min MHz:         800.0000
BogoMIPS:            5200.00
Virtualization:      VT-x
L1d cache:           48K
L1i cache:           32K
L2 cache:            2048K
L3 cache:            61440K
NUMA node0 CPU(s):   0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46
NUMA node1 CPU(s):   1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47
$ exit
</pre>


### adroit-h11g2

`adroit-h11g2` has 4 NVIDIA A100 GPUs with 40 GB of memory per GPU. To connect to this node use:

```
$ salloc --nodes=1 --ntasks=1 --mem=4G --time=00:05:00 --gres=gpu:1 --nodelist=adroit-h11g2
```

Below is information about the A100 GPUs:

```
Using a NVIDIA A100-PCIE-40GB GPU.
  CUDADevice with properties:

                      Name: 'NVIDIA A100-PCIE-40GB'
                     Index: 1
         ComputeCapability: '8.0'
            SupportsDouble: 1
             DriverVersion: 11.7000
            ToolkitVersion: 11.2000
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 4.2351e+10
           AvailableMemory: 4.1703e+10
       MultiprocessorCount: 108
              ClockRateKHz: 1410000
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 0
          CanMapHostMemory: 1
           DeviceSupported: 1
           DeviceAvailable: 1
            DeviceSelected: 1
```

Below is information about the CPUs:

```
$ lscpu | grep -v Flags
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              48
On-line CPU(s) list: 0-47
Thread(s) per core:  1
Core(s) per socket:  24
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6
Model:               106
Model name:          Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz
Stepping:            6
CPU MHz:             3499.996
CPU max MHz:         3500.0000
CPU min MHz:         800.0000
BogoMIPS:            5600.00
L1d cache:           48K
L1i cache:           32K
L2 cache:            1280K
L3 cache:            36864K
NUMA node0 CPU(s):   0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46
NUMA node1 CPU(s):   1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47
```

Add these lines to your Slurm script to explicitly use an A100 GPU:

```
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
```

To see a wealth of information about the GPUs use:

```
$ nvidia-smi -q
```

### adroit-h11g3

This node offers the older V100 GPUs.

### Compute Capability and Building Optimized Codes

Some software will only run on a GPU of a given compute capability. To find these values for a given NVIDIA Telsa card see [this page](https://en.wikipedia.org/wiki/Nvidia_Tesla). The compute capability of the A100's on Della is 8.0. For various build systems this translates to `sm_80`.

The following is from `$ nvcc --help` after loading a `cudatoolkit` module:

```
Options for steering GPU code generation.
=========================================

--gpu-architecture <arch>                       (-arch)                         
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
        -arch=all         build for all supported architectures (sm_*), and add PTX
        for the highest major architecture to the generated code.
        -arch=all-major   build for just supported major versions (sm_*0), plus the
        earliest supported, and add PTX for the highest major architecture to the
        generated code.
        -arch=native      build for all architectures (sm_*) on the current system
        Note: -arch=native, -arch=all, -arch=all-major cannot be used with the -code
        option, but can be used with -gencode options
        Note: the values compute_30, compute_32, compute_35, compute_37, compute_50,
        sm_30, sm_32, sm_35, sm_37 and sm_50 are deprecated and may be removed in
        a future release.
        Allowed values for this option:  'all','all-major','compute_35','compute_37',
        'compute_50','compute_52','compute_53','compute_60','compute_61','compute_62',
        'compute_70','compute_72','compute_75','compute_80','compute_86','compute_87',
        'lto_35','lto_37','lto_50','lto_52','lto_53','lto_60','lto_61','lto_62',
        'lto_70','lto_72','lto_75','lto_80','lto_86','lto_87','native','sm_35','sm_37',
        'sm_50','sm_52','sm_53','sm_60','sm_61','sm_62','sm_70','sm_72','sm_75',
        'sm_80','sm_86','sm_87'.
```

Hence, a starting point for optimization flags for the A100 GPUs on Della and Adroit:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_80 -o myapp myapp.cu
```

For the V100's on Adroit or Traverse would be:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_70 -o myapp myapp.cu
```

## TigerGPU

TigerGPU was composed of 80 Intel Broadwell nodes each with 4 NVIDIA P100 GPUs. See the P100 [technical specs](https://www.techpowerup.com/gpu-specs/tesla-p100-pcie-16-gb.c2888) or buy on [Amazon](https://www.amazon.com/NVIDIA-Tesla-Passive-Accelerator-900-2H400-0000-000/dp/B0792FXS2S/ref=sr_1_1?keywords=nvidia+p100&qid=1572465106&sr=8-1). Each GPU had 56 streaming multiprocessors (SM) and 64 CUDA FP32 cores per SM.

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

The [Traverse cluster](https://www.princeton.edu/news/2019/10/07/princetons-new-supercomputer-traverse-accelerate-scientific-discovery-fusion) consists of 46 IBM Power9 nodes with 4 NVIDIA V100 GPUs. It is a smaller version of the [Sierra](https://en.wikipedia.org/wiki/Sierra_(supercomputer)) supercomputer. The GPUs on Traverse have 32 GB of memory each and a clock rate of 1.29 GHz. Each GPU has 80 streaming multiprocessors (SM) and 64 CUDA cores per SM (and 8 Tensor Cores per SM).

Additional info:

```
$ ssh <NetID>@traverse.princeton.edu
$ salloc --nodes=1 --ntasks=1 --mem=4G --time=00:10:00 --gres=gpu:1
$ nvidia-smi -q
```

## Comparison of GPU Resources

|   Cluster  | Number of Nodes | GPUs per Node | NVIDIA GPU Model  | Number of FP32 Cores| SM Count | GPU Memory (GB) |
|:----------:|:----------:|:---------:|:-------:|:-------:|:-------:|:-------:|
| Adroit     |      2           |     4         |  V100            | 5120   | 80  | 32 |
| Adroit     |      1           |     4         |  A100            | 6912   | 108  | 40 |     
| Della      |     70           |     4         |  A100            | 6912   | 108  | 80 |
| Della      |     20           |     2         |  A100            | 6912   | 108  | 40 |
| Stella     |     6            |     2         |  A100            | 6912   | 108  | 40 |
| TigerGPU   |     80           |     4         |  P100            | 3584   | 56  | 16 |
| Traverse   |     46           |     4         |  V100            | 5120   | 80  | 32 | 

SM is streaming multiprocessor. Note that the V100 GPUs have 640 [Tensor Cores](https://devblogs.nvidia.com/cuda-9-features-revealed/) (8 per SM) where half-precision Warp Matrix-Matrix and Accumulate (WMMA) operations can be carried out. That is, each core can perform a 4x4 matrix-matrix multiply and add the result to a third matrix. There are differences between the V100 node on Adroit and the Traverse nodes (see [PCIe versus SXM2](https://www.nextplatform.com/micro-site-content/achieving-maximum-compute-throughput-pcie-vs-sxm2/)).


## GPU Hackathon at Princeton

The next hackathon will take place in June of 2023. This is a great opportunity to get help from experts in porting your code to a GPU. Or you can participate as a mentor and help a team rework their code. See the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for details.
