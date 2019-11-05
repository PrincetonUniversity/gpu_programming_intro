# What is a GPU?

A GPU, or Graphics Processing Unit, is an electronic device originally designed for manipulating the images that appear on a computer monitor.  However GPUs have become widely used for accelerating computation in various fields including image processing and machine learning.

Relative to the CPU, GPUs have a far greater number of processing cores with slower clock speeds. Within a block of threads, each thread carries out the same operation on a different piece of data. This is the SIMT paradigm (single instruction, multiple threads). GPUs tend to have less memory that what is available to a CPU. The P100 GPUs on TigerGPU have 16 GB. This is an important consideration in designing algorithms and pipelines.

Many of the fastest supercomputers in the world use GPUs (see [Top 500](https://www.top500.org/lists/2019/06/)).

While Princeton relies on NVIDIA, the GPU market landscape changed in May 2019 when it was announced that Frontier, what is expected to be the first exascale supercomputer in the US, would be based on [AMD GPUs](https://www.hpcwire.com/2019/05/07/cray-amd-exascale-frontier-at-oak-ridge/) and CPUs.

![cpu-vs-gpu](http://blog.itvce.com/wp-content/uploads/2016/03/032216_1532_DustFreeNVI2.png)

# Overview of using a GPU

This is the essence of how every GPU is used as an accelerator:

+ Copy data from the CPU (host) to the GPU (device)

+ Launch a kernel to carry out computations on the GPU

+ Copy data from the GPU (device) back to the CPU (host)

![gpu-overview](https://blogandcode.files.wordpress.com/2013/12/cudac-1.jpeg?w=597&h=372)

# What GPU resources does Princeton have?

## Adroit

There are 2 GPU nodes on Adroit: `adroit-h11g1` and `adroit-h11g4`

<pre>
$ ssh &lt;NetID&gt;@adroit.princeton.edu
$ snodes

HOSTNAMES     STATE    CPUS S:C:T    CPUS(A/I/O/T)   CPU_LOAD MEMORY   GRES     PARTITION          AVAIL_FEATURES
adroit-01     idle     20   2:10:1   0/20/0/20       0.01     128000   (null)   class              ivy
adroit-02     idle     20   2:10:1   0/20/0/20       0.01     64000    (null)   class              ivy
adroit-03     idle     20   2:10:1   0/20/0/20       0.01     64000    (null)   class              ivy
adroit-04     idle     20   2:10:1   0/20/0/20       0.01     64000    (null)   class              ivy
adroit-05     idle     20   2:10:1   0/20/0/20       0.01     64000    (null)   class              ivy
adroit-06     idle     20   2:10:1   0/20/0/20       0.01     64000    (null)   class              ivy
adroit-07     idle     20   2:10:1   0/20/0/20       0.01     64000    (null)   class              ivy
adroit-08     mix      32   2:16:1   30/2/0/32       19.94    384000   (null)   all*               skylake
adroit-09     alloc    32   2:16:1   32/0/0/32       57.28    384000   (null)   all*               skylake
adroit-10     mix      32   2:16:1   31/1/0/32       15.63    384000   (null)   all*               skylake
adroit-11     mix      32   2:16:1   30/2/0/32       1.66     384000   (null)   all*               skylake
adroit-12     alloc    32   2:16:1   32/0/0/32       59.14    384000   (null)   all*               skylake
adroit-13     mix      32   2:16:1   31/1/0/32       27.33    384000   (null)   all*               skylake
adroit-14     alloc    32   2:16:1   32/0/0/32       12.19    384000   (null)   all*               skylake
adroit-15     alloc    32   2:16:1   32/0/0/32       31.51    384000   (null)   all*               skylake
adroit-16     alloc    32   2:16:1   32/0/0/32       31.68    384000   (null)   all*               skylake
<b>adroit-h11g1  mix      40   2:20:1   33/7/0/40       29.78    770000   gpu:tesl gpu                (null)</b>
<b>adroit-h11g4  mix      16   2:8:1    1/15/0/16       0.93     64000    gpu:tesl gpu                (null)</b>
</pre>

### `adroit-h11g1`

This node has 4 NVIDIA V100 GPUs with 32 GB of memory each. See the specs for the [V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-32-gb.c3184) or consider buying on [Amazon](https://www.amazon.com/NVIDIA-Tesla-Volta-Accelerator-Graphics/dp/B07JVNHFFX/ref=sr_1_2?keywords=nvidia+v100&qid=1572464893&sr=8-2).

Add this line to your Slurm script to use a V100 GPUs:

```
#SBATCH --gres=gpu:tesla_v100:1
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

### `adroit-h11g4`

`adroit-h11g4` has 2 NVIDIA K40c GPUs with 12 GB of memory per GPU. View the technical specifications for the [K40c](https://www.techpowerup.com/gpu-specs/tesla-k40c.c2505) or buy this GPU on [Amazon](https://www.amazon.com/NVIDIA-Tesla-K40c-computing-processor/dp/B06VSWDH15/ref=sr_1_3?keywords=nvidia+k40c&qid=1572468693&sr=8-3).

Add this line to your Slurm script to use a K40c GPU:

```
#SBATCH --gres=gpu:tesla_k40c:1
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

Hence, a starting point for optimization flags on the V100's on Adroit or Traverse would be:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_70 --gpu-code=sm_70 -o myapp myapp.cu
```

For the P100 GPUs on TigerGPU:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_60 --gpu-code=sm_60 -o myapp myapp.cu
```

And for the K40c GPUs on Adroit:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_35 --gpu-code=sm_35 -o myapp myapp.cu
```

## TigerGPU

TigerGPU has 80 Intel Broadwell nodes each with four NVIDIA P100 GPUs. See the P100 [technical specs](https://www.techpowerup.com/gpu-specs/tesla-p100-pcie-16-gb.c2888) or buy on [Amazon](https://www.amazon.com/NVIDIA-Tesla-Passive-Accelerator-900-2H400-0000-000/dp/B0792FXS2S/ref=sr_1_1?keywords=nvidia+p100&qid=1572465106&sr=8-1).

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

Below is the output of the `nvidia-smi` command:

```
$ hostname
tiger-i20g7

$ nvidia-smi
Sun Oct 27 22:56:23 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  On   | 00000000:03:00.0 Off |                    0 |
| N/A   48C    P0    80W / 250W |  15757MiB / 16280MiB |     74%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla P100-PCIE...  On   | 00000000:04:00.0 Off |                    0 |
| N/A   37C    P0    34W / 250W |  15753MiB / 16280MiB |      9%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla P100-PCIE...  On   | 00000000:82:00.0 Off |                    0 |
| N/A   45C    P0    32W / 250W |  15719MiB / 16280MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla P100-PCIE...  On   | 00000000:83:00.0 Off |                    0 |
| N/A   49C    P0    79W / 250W |  15757MiB / 16280MiB |     74%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     32636      C   python                                     15747MiB |
|    1     29633      C   python                                     15743MiB |
|    2     19218      C   python                                     15709MiB |
|    3       419      C   python                                     15747MiB |
+-----------------------------------------------------------------------------+
```

Below is the output of the `gpustat` command:

```
$ gpustat
tiger-i20g7  Sun Oct 27 22:56:31 2019
[0] Tesla P100-PCIE-16GB | 48'C,  55 % | 15757 / 16280 MB | eham(15747M)
[1] Tesla P100-PCIE-16GB | 37'C,   9 % | 15753 / 16280 MB | junyuli(15743M)
[2] Tesla P100-PCIE-16GB | 51'C,  94 % | 15719 / 16280 MB | oahmed(15709M)
[3] Tesla P100-PCIE-16GB | 48'C,  75 % | 15757 / 16280 MB | eham(15747M)
```

## Traverse

This new cluster consists of 46 IBM Power9 nodes with 4 NVIDIA V100 GPUs. It is a smaller version of the [Sierra](https://en.wikipedia.org/wiki/Sierra_(supercomputer)) supercomputer. The GPUs on Traverse have 32 GB of memory each and a clock rate of 1.53 GHz.

Additional info:

```
$ ssh traverse-k02g3
$ nvidia-smi -q

==============NVSMI LOG==============

Timestamp                           : Wed Oct 30 09:52:39 2019
Driver Version                      : 418.67
CUDA Version                        : 10.1

Attached GPUs                       : 4
GPU 00000004:04:00.0
    Product Name                    : Tesla V100-SXM2-32GB
    Product Brand                   : Tesla
    Display Mode                    : Enabled
    Display Active                  : Disabled
    Persistence Mode                : Enabled
    Accounting Mode                 : Disabled
    Accounting Mode Buffer Size     : 4000
    Driver Model
        Current                     : N/A
        Pending                     : N/A
    Serial Number                   : 0561519020531
    GPU UUID                        : GPU-fd2f54d7-d5ed-ea23-b43e-eb240a7c7193
    Minor Number                    : 0
    VBIOS Version                   : 88.00.80.00.01
    MultiGPU Board                  : No
    Board ID                        : 0x40400
    GPU Part Number                 : 900-2G503-0430-000
    Inforom Version
        Image Version               : G503.0203.00.05
        OEM Object                  : 1.1
        ECC Object                  : 5.0
        Power Management Object     : N/A
    GPU Operation Mode
        Current                     : N/A
        Pending                     : N/A
    GPU Virtualization Mode
        Virtualization mode         : None
    IBMNPU
        Relaxed Ordering Mode       : Disabled
    PCI
        Bus                         : 0x04
        Device                      : 0x00
        Domain                      : 0x0004
        Device Id                   : 0x1DB510DE
        Bus Id                      : 00000004:04:00.0
        Sub System Id               : 0x124910DE
        GPU Link Info
            PCIe Generation
                Max                 : 3
                Current             : 3
            Link Width
                Max                 : 16x
                Current             : 2x
        Bridge Chip
            Type                    : N/A
            Firmware                : N/A
        Replays Since Reset         : 0
        Replay Number Rollovers     : 0
        Tx Throughput               : 0 KB/s
        Rx Throughput               : 0 KB/s
    Fan Speed                       : N/A
    Performance State               : P0
    Clocks Throttle Reasons
        Idle                        : Active
        Applications Clocks Setting : Not Active
        SW Power Cap                : Not Active
        HW Slowdown                 : Not Active
            HW Thermal Slowdown     : Not Active
            HW Power Brake Slowdown : Not Active
        Sync Boost                  : Not Active
        SW Thermal Slowdown         : Not Active
        Display Clock Setting       : Not Active
    FB Memory Usage
        Total                       : 32480 MiB
        Used                        : 10 MiB
        Free                        : 32470 MiB
    BAR1 Memory Usage
        Total                       : 32768 MiB
        Used                        : 0 MiB
        Free                        : 32768 MiB
    Compute Mode                    : Default
    Utilization
        Gpu                         : 0 %
        Memory                      : 0 %
        Encoder                     : 0 %
        Decoder                     : 0 %
    Encoder Stats
        Active Sessions             : 0
        Average FPS                 : 0
        Average Latency             : 0
    FBC Stats
        Active Sessions             : 0
        Average FPS                 : 0
        Average Latency             : 0
    Ecc Mode
        Current                     : Enabled
        Pending                     : Enabled
    ECC Errors
        Volatile
            Single Bit            
                Device Memory       : 0
                Register File       : 0
                L1 Cache            : 0
                L2 Cache            : 0
                Texture Memory      : N/A
                Texture Shared      : N/A
                CBU                 : N/A
                Total               : 0
            Double Bit            
                Device Memory       : 0
                Register File       : 0
                L1 Cache            : 0
                L2 Cache            : 0
                Texture Memory      : N/A
                Texture Shared      : N/A
                CBU                 : 0
                Total               : 0
        Aggregate
            Single Bit            
                Device Memory       : 0
                Register File       : 0
                L1 Cache            : 0
                L2 Cache            : 0
                Texture Memory      : N/A
                Texture Shared      : N/A
                CBU                 : N/A
                Total               : 0
            Double Bit            
                Device Memory       : 0
                Register File       : 0
                L1 Cache            : 0
                L2 Cache            : 0
                Texture Memory      : N/A
                Texture Shared      : N/A
                CBU                 : 0
                Total               : 0
    Retired Pages
        Single Bit ECC              : 0
        Double Bit ECC              : 0
        Pending                     : No
    Temperature
        GPU Current Temp            : 38 C
        GPU Shutdown Temp           : 90 C
        GPU Slowdown Temp           : 87 C
        GPU Max Operating Temp      : 83 C
        Memory Current Temp         : 34 C
        Memory Max Operating Temp   : 85 C
    Power Readings
        Power Management            : Supported
        Power Draw                  : 39.78 W
        Power Limit                 : 300.00 W
        Default Power Limit         : 300.00 W
        Enforced Power Limit        : 300.00 W
        Min Power Limit             : 150.00 W
        Max Power Limit             : 300.00 W
    Clocks
        Graphics                    : 135 MHz
        SM                          : 135 MHz
        Memory                      : 877 MHz
        Video                       : 555 MHz
    Applications Clocks
        Graphics                    : 1290 MHz
        Memory                      : 877 MHz
    Default Applications Clocks
        Graphics                    : 1290 MHz
        Memory                      : 877 MHz
    Max Clocks
        Graphics                    : 1530 MHz
        SM                          : 1530 MHz
        Memory                      : 877 MHz
        Video                       : 1372 MHz
    Max Customer Boost Clocks
        Graphics                    : 1530 MHz
    Clock Policy
        Auto Boost                  : N/A
        Auto Boost Default          : N/A
    Processes                       : None
...
```

## Should you buy an NVIDIA V100 for your research use?

Below is the answer from one of the sys admins:

> The V100 will kill you.  They are about $20k retail.  For a workstation 
> I'd think the best nVidia GPU you could purchase would be enough to get 
> started.  Like the GeForce line.  It supports Cuda but doesn't have the 
> memory of a V100.  But for development and testing it should be "good 
> enough" for ramping up to the bigger V100s.

For a cost analysis by MicroWay see [this page](https://www.microway.com/hpc-tech-tips/nvidia-tesla-v100-price-analysis/).


## NVIDIA GPU Hackathon at Princeton

If you would like to serve as a junior GPU programming mentor to help Princeton researchers port their CPU codes to GPUs then please join. The hackathon will run from June 8-12, 2020.
