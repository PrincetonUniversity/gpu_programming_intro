# What is a GPU?

A GPU, or Graphics Processing Unit, is an electronic device originally designed for manipulating the images that appear on a computer monitor. However, beginning in 2006 with NVIDIA CUDA, GPUs have become widely used for accelerating computation in numerous fields including image processing and machine learning.

Relative to the CPU, GPUs have a far greater number of processing cores. Within a block of threads called a warp (NVIDIA), each thread carries out the same operation on a different piece of data. This is the SIMT paradigm (single instruction, multiple threads). In data centers, GPUs tend to have much less memory than what is available on a CPU. For instance, the H100 GPUs on Della have 80 GB compared to 1500 GB available to the CPU cores. This is an important consideration when designing algorithms and running jobs. Furthermore, GPUs are intended for highly parallel algorithms which allow for latency hiding. The CPU can out-perform a GPU on algorithms that are not highly parallelizable such as those that rely on data caching and flow control (e.g., "if" statements).

NVIDIA has been the leading player in GPUs for HPC and AI. However, the GPU market landscape changed in May 2019 when the US DoE announced that [Frontier](https://www.hpcwire.com/2019/05/07/cray-amd-exascale-frontier-at-oak-ridge/), the first exascale supercomputer in the US, would be based on [AMD GPUs](https://www.amd.com/en/products/accelerators/instinct.html) and CPUs. Princeton has a two [MI210 GPUs](https://researchcomputing.princeton.edu/amd-mi100-gpu-testing) which you can use for testing. Intel is also a GPU producer with the [Aurora supercomputer](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) being an example.

Many of the fastest supercomputers in the world use GPUs (see [Top 500](https://top500.org/lists/top500/2025/06/)). How many of the top 10 supercomputers use GPUs?

All laptops have a GPU for graphics. It is becoming standard for a laptop to have a second GPU dedicated for compute (see the latest [MacBook Pro](https://www.apple.com/macbook-pro/)).

![cpu-vs-gpu](http://blog.itvce.com/wp-content/uploads/2016/03/032216_1532_DustFreeNVI2.png)

The image below emphasizes the cache sizes and flow control:

![cache_flow_control](https://tigress-web.princeton.edu/~jdh4/gpu-devotes-more-transistors-to-data-processing.png)

Like a CPU, a GPU has a hierarchical structure with respect to both the execution units and memory. A warp is a unit of 32 threads. NVIDIA GPUs impose a limit of 1024 threads per block. Some integral number of warps are grouped into a streaming multiprocessor (SM). There are tens of SMs per GPU. Each thread has its own memory. There is limited shared memory between a block of threads. And, finally, there is the global memory which is accessible to each grid or collection of blocks.

![ampere](https://developer-blogs.nvidia.com/wp-content/uploads/2022/03/H100-Streaming-Multiprocessor-SM-625x869.png)

The figure above is a diagram of a streaming multiprocessor (SM) for the [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/). The H100 is composed of up to 132 SMs.

# Princeton Language and Intelligence

The university spent $9.6M on a new [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) cluster for research involving large AI models. The cluster provides 42 nodes with 8 GPUs per node. The H100 GPU is optimized for training transformer models. [Learn more](https://pli.princeton.edu/about-pli/directors-message) about this.

# Overview of using a GPU

This is the essence of how every GPU is used as an accelerator for compute:

+ Copy data from the CPU (host) to the GPU (device)

+ Launch a kernel to carry out computations on the GPU

+ Copy data from the GPU (device) back to the CPU (host)

![gpu-overview](https://tigress-web.princeton.edu/~jdh4/gpu_as_accelerator_to_cpu_diagram.png)

The diagram above and the accompanying pseudocode present a simplified view of how GPUs are used in scientific computing. To fully understand how things work you will need to learn more about memory cache, interconnects, CUDA streams and much more.

[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) on Traverse enables fast CPU-to-GPU and GPU-to-GPU data transfers with a peak rate of 75 GB/s per direction.

Given the significant performance penalty for moving data between the CPU and GPU, it is natural to work toward "unifying" the CPU and GPU. For instance, read about the [NVIDIA Grace Hopper Superchip](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/).

# What GPU resources does Princeton have?

See the "Hardware Resources" on the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for a complete list.

## Adroit

There are 3 GPU nodes on Adroit: `adroit-h11g1`, `adroit-h11g2` and `adroit-h11g3`

<pre>
$ ssh &lt;NetID&gt;@adroit.princeton.edu
$ snodes
HOSTNAMES      STATE  CPUS S:C:T  CPUS(A/I/O/T) CPU_LOAD MEMORY  PARTITION  AVAIL_FEATURES
adroit-08      alloc  32   2:16:1 32/0/0/32     1.27     384000  class      skylake,intel
adroit-09      alloc  32   2:16:1 32/0/0/32     0.75     384000  class      skylake,intel
adroit-10      alloc  32   2:16:1 32/0/0/32     0.63     384000  class      skylake,intel
adroit-11      mix    32   2:16:1 29/3/0/32     0.28     384000  class      skylake,intel
adroit-12      mix    32   2:16:1 16/16/0/32    0.28     384000  class      skylake,intel
adroit-13      mix    32   2:16:1 25/7/0/32     0.22     384000  all*       skylake,intel
adroit-13      mix    32   2:16:1 25/7/0/32     0.22     384000  class      skylake,intel
adroit-14      alloc  32   2:16:1 32/0/0/32     32.29    384000  all*       skylake,intel
adroit-14      alloc  32   2:16:1 32/0/0/32     32.29    384000  class      skylake,intel
adroit-15      mix    32   2:16:1 22/10/0/32    9.68     384000  all*       skylake,intel
adroit-15      mix    32   2:16:1 22/10/0/32    9.68     384000  class      skylake,intel
adroit-16      alloc  32   2:16:1 32/0/0/32     24.13    384000  all*       skylake,intel
adroit-16      alloc  32   2:16:1 32/0/0/32     24.13    384000  class      skylake,intel
adroit-h11g1   plnd   48   2:24:1 0/48/0/48     0.00     1000000 gpu        a100,intel,gpu80
adroit-h11g2   plnd   48   2:24:1 0/48/0/48     0.76     1000000 gpu        a100,intel
adroit-h11g3   mix    56   4:14:1 5/51/0/56     1.05     760000  gpu        v100,intel
adroit-h11n1   idle   128  2:64:1 0/128/0/128   0.00     256000  class      amd,rome
adroit-h11n2   alloc  64   2:32:1 64/0/0/64     49.07    500000  all*       intel,ice
adroit-h11n3   mix    64   2:32:1 50/14/0/64    40.54    500000  all*       intel,ice
adroit-h11n4   mix    64   2:32:1 48/16/0/64    40.33    500000  all*       intel,ice
adroit-h11n5   mix    64   2:32:1 32/32/0/64    32.94    500000  all*       intel,ice
adroit-h11n6   mix    64   2:32:1 62/2/0/64     38.95    500000  all*       intel,ice
</pre>

To only see the GPU nodes:

<pre>
$ shownodes -p gpu
NODELIST      STATE      FREE/TOTAL CPUs  CPU_LOAD  AVAIL/TOTAL MEMORY  FREE/TOTAL GPUs          FEATURES
adroit-h11g1  planned              48/48      0.00   1000000/1000000MB  4/4 nvidia_a100  a100,intel,gpu80
adroit-h11g2  planned              48/48      0.76   1000000/1000000MB      8/8 3g.20gb        a100,intel
adroit-h11g3  mixed                51/56      1.05     736960/760000MB   0/4 tesla_v100        v100,intel
</pre>

### adroit-h11g1

This node has 4 NVIDIA A100 GPUs with 80 GB of memory each. Each A100 GPU has 108 streaming multiprocessors (SM) and 64 FP32 CUDA cores per SM.

Here is some information about the A100 GPUs on this node:

```
  CUDADevice with properties:

                      Name: 'NVIDIA A100 80GB PCIe'
                     Index: 1
         ComputeCapability: '8.0'
            SupportsDouble: 1
             DriverVersion: 12.2000
            ToolkitVersion: 11.2000
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [2.1475e+09 65535 65535]
                 SIMDWidth: 32
               TotalMemory: 8.5175e+10
           AvailableMemory: 8.4519e+10
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

Here is infomation about the CPUs on this node:

<pre>
$ ssh &lt;NetID&gt;@adroit.princeton.edu
$ salloc --nodes=1 --ntasks=1 --mem=4G --time=00:05:00 --gres=gpu:1 --constraint=gpu80 --reservation=gpuprimer
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

`adroit-h11g2` has 4 NVIDIA A100 GPUs with 40 GB of memory per GPU. The 4 GPUs have been divided into 8 less powerful GPUs with 20 GB of memory each. To connect to this node use:

```
$ salloc --nodes=1 --ntasks=1 --mem=4G --time=00:05:00 --gres=gpu:1 --nodelist=adroit-h11g2 --reservation=gpuprimer
```

Below is information about the A100 GPUs:

```
$ nvidia-smi -a
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

See the necessary Slurm directives to [run on specific GPUs](https://researchcomputing.princeton.edu/systems/adroit#gpus) on Adroit.

To see a wealth of information about the GPUs use:

```
$ nvidia-smi -q | less
```

### adroit-h11g3

This node offers the older V100 GPUs.

### Grace Hopper Superchip

See the [Grace Hopper Superchip webpage](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/) by NVIDIA. Here is a schematic diagram of the superchip:

![grace](https://developer-blogs.nvidia.com/wp-content/uploads/2022/11/grace-hopper-overview.png)

```
aturing@della-gh:~$ nvidia-smi -a

==============NVSMI LOG==============

Timestamp                                 : Mon Apr 22 11:24:41 2024
Driver Version                            : 545.23.08
CUDA Version                              : 12.3

Attached GPUs                             : 1
GPU 00000009:01:00.0
    Product Name                          : GH200 480GB
    Product Brand                         : NVIDIA
    Product Architecture                  : Hopper
    Display Mode                          : Disabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    Addressing Mode                       : ATS
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
...
```

The CPU on the GH Superchip:

```
jdh4@della-gh:~$ lscpu
Architecture:           aarch64
  CPU op-mode(s):       64-bit
  Byte Order:           Little Endian
CPU(s):                 72
  On-line CPU(s) list:  0-71
Vendor ID:              ARM
  Model name:           Neoverse-V2
    Model:              0
    Thread(s) per core: 1
    Core(s) per socket: 72
    Socket(s):          1
    Stepping:           r0p0
    Frequency boost:    disabled
    CPU max MHz:        3510.0000
    CPU min MHz:        81.0000
    BogoMIPS:           2000.00
    Flags:              fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm di
                        t uscat ilrcpc flagm ssbs sb dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh
Caches (sum of all):    
  L1d:                  4.5 MiB (72 instances)
  L1i:                  4.5 MiB (72 instances)
  L2:                   72 MiB (72 instances)
  L3:                   114 MiB (1 instance)
NUMA:                   
  NUMA node(s):         9
  NUMA node0 CPU(s):    0-71
  NUMA node1 CPU(s):    
  NUMA node2 CPU(s):    
  NUMA node3 CPU(s):    
  NUMA node4 CPU(s):    
  NUMA node5 CPU(s):    
  NUMA node6 CPU(s):    
  NUMA node7 CPU(s):    
  NUMA node8 CPU(s):    
Vulnerabilities:        
  Gather data sampling: Not affected
  Itlb multihit:        Not affected
  L1tf:                 Not affected
  Mds:                  Not affected
  Meltdown:             Not affected
  Mmio stale data:      Not affected
  Retbleed:             Not affected
  Spec rstack overflow: Not affected
  Spec store bypass:    Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:           Mitigation; __user pointer sanitization
  Spectre v2:           Not affected
  Srbds:                Not affected
  Tsx async abort:      Not affected
```

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

For the H100 GPUs on Della:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_90 -o myapp myapp.cu
```

## Comparison of GPU Resources

|   Cluster  | Number of Nodes | GPUs per Node | NVIDIA GPU Model  | Number of FP32 Cores| SM Count | GPU Memory (GB) |
|:----------:|:----------:|:---------:|:-------:|:-------:|:-------:|:-------:|
| Adroit     |      1           |     4         |  A100            | 6912   | 108  | 80 |
| Adroit     |      1           |     8         |  A100            | --   | --  | 20 |
| Adroit     |      1           |     4         |  V100            | 5120   | 80  | 32 |    
| Della      |     42           |     8         |  H100            | 16896  | 132 | 80 |
| Della      |     69           |     4         |  A100            | 6912   | 108  | 80 |
| Della      |     20           |     2         |  A100            | 6912   | 108  | 40 |
| Della      |     2            |    28         |  A100            | --     | --   | 10 |  
| Stellar    |     6            |     2         |  A100            | 6912   | 108  | 40 |
| Stellar    |     1            |     8         |  A100            | 6912   | 108  | 40 |
| Tiger      |     1            |     8         |  H200            | 16896  | 144  | 141 |
| Tiger      |     12           |     4         |  H100            | 14592  | 132  | 80 |
| Tiger      |     40           |     1         |  L40S            | 18176  | 142  | 48 |

SM is streaming multiprocessor. Note that the V100 GPUs have 640 [Tensor Cores](https://devblogs.nvidia.com/cuda-9-features-revealed/) (8 per SM) where half-precision Warp Matrix-Matrix and Accumulate (WMMA) operations can be carried out. That is, each core can perform a 4x4 matrix-matrix multiply and add the result to a third matrix.

## GPU Hackathon at Princeton

The previous hackathon took place in [June of 2025](https://www.openhackathons.org/s/siteevent/a0CUP00000rwmKa2AI/se000356). These events are a great opportunity to get help from experts in porting your code to a GPU. Or you can participate as a mentor and help a team rework their code. See the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for details.
