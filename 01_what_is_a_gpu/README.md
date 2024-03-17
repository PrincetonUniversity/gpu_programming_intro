# What is a GPU?

A GPU, or Graphics Processing Unit, is an electronic device originally designed for manipulating the images that appear on a computer monitor. However, beginning around 2006 with NVIDIA CUDA, GPUs have become widely used for accelerating computation in various fields including image processing and machine learning.

Relative to the CPU, GPUs have a far greater number of processing cores but with slower clock speeds. Within a block of threads called a warp (NVIDIA), each thread carries out the same operation on a different piece of data. This is the SIMT paradigm (single instruction, multiple threads). GPUs tend to have much less memory than what is available on a CPU. For instance, the A100 GPUs on Della have 80 GB compared to 1000 GB available to the CPU cores. This is an important consideration when designing algorithms and running jobs. Furthermore, GPUs are intended for highly parallel algorithms. The CPU can often out-perform a GPU on algorithms that are not highly parallelizable such as those that rely on data caching and flow control (e.g., "if" statements).

Many of the fastest supercomputers in the world use GPUs (see [Top 500](https://www.top500.org/lists/top500/2023/11/)). How many of the top 10 supercomputers use GPUs?

NVIDIA has been the leading player in GPUs for HPC. However, the GPU market landscape changed in May 2019 when the US DoE announced that Frontier, the first exascale supercomputer in the US, would be based on [AMD GPUs](https://www.hpcwire.com/2019/05/07/cray-amd-exascale-frontier-at-oak-ridge/) and CPUs. Princeton has a two [MI210 GPUs](https://researchcomputing.princeton.edu/amd-mi100-gpu-testing) which you can use for testing. Intel will soon be a new player when the [Aurora supercomputer](https://en.wikipedia.org/wiki/Aurora_(supercomputer)) is completed.

All laptops have a GPU for graphics. It is becoming standard for a laptop to have a second GPU dedicated for compute (see the latest [MacBook Pro](https://www.apple.com/macbook-pro/)).

![cpu-vs-gpu](http://blog.itvce.com/wp-content/uploads/2016/03/032216_1532_DustFreeNVI2.png)

The image below emphasizes the cache sizes and flow control:

![cache_flow_control](https://tigress-web.princeton.edu/~jdh4/gpu-devotes-more-transistors-to-data-processing.png)

Like a CPU, a GPU has a hierarchical structure with respect to both the execution units and memory. A warp is a unit of 32 threads. NVIDIA GPUs impose a limit of 1024 threads per block. Some integral number of warps are grouped into a streaming multiprocessor (SM). There are tens of SMs per GPU. Each thread has its own memory. There is limited shared memory between a block of threads. And, finally, there is the global memory which is accessible to each grid or collection of blocks.

![ampere](https://developer-blogs.nvidia.com/wp-content/uploads/2021/guc/raD52-V3yZtQ3WzOE0Cvzvt8icgGHKXPpN2PS_5MMyZLJrVxgMtLN4r2S2kp5jYI9zrA2e0Y8vAfpZia669pbIog2U9ZKdJmQ8oSBjof6gc4IrhmorT2Rr-YopMlOf1aoU3tbn5Q.png)

The figure above is a diagram of a streaming multiprocessor (SM) for the [NVIDIA A100 GPU](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/). The A100 is composed of 108 SMs.

# Princeton Language and Intelligence

The university spent $9.6M on a new [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) cluster for research involving large AI models. The cluster will provide 37 nodes with 8 GPUs per node. The H100 GPU is optimized for training transformer models. [Learn more](https://pli.princeton.edu/about-pli/directors-message) about this.

# Overview of using a GPU

This is the essence of how every GPU is used as an accelerator for compute:

+ Copy data from the CPU (host) to the GPU (device)

+ Launch a kernel to carry out computations on the GPU

+ Copy data from the GPU (device) back to the CPU (host)

![gpu-overview](https://tigress-web.princeton.edu/~jdh4/gpu_as_accelerator_to_cpu_diagram.png)

The diagram above and the accompanying pseudocode present a simplified view of how GPUs are used in scientific computing. To fully understand how things work you will need to learn more about memory cache, interconnects, CUDA streams and much more.

[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/) on Traverse enables fast CPU-to-GPU and GPU-to-GPU data transfers with a peak rate of 75 GB/s per direction. Della has this fast GPU-GPU interconnect on each pair of GPUs on 70 of the 90 GPU nodes.

Given the significant performance penalty for moving data between the CPU and GPU, it is natural to work toward "unifying" the CPU and GPU. For instance, read about the [NVIDIA Grace Hopper Superchip](https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/).

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

`adroit-h11g2` has 4 NVIDIA A100 GPUs with 40 GB of memory per GPU. To connect to this node use:

```
$ salloc --nodes=1 --ntasks=1 --mem=4G --time=00:05:00 --gres=gpu:1 --nodelist=adroit-h11g2 --reservation=gpuprimer
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

See the necessary Slurm directives to [run on specific GPUs](https://researchcomputing.princeton.edu/systems/adroit#gpus) on Adroit.

To see a wealth of information about the GPUs use:

```
$ nvidia-smi -q | less
```

### adroit-h11g3

This node offers the older V100 GPUs.

### PLI Nodes

```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              96
On-line CPU(s) list: 0-95
Thread(s) per core:  1
Core(s) per socket:  48
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6
Model:               143
Model name:          Intel(R) Xeon(R) Platinum 8468
Stepping:            8
CPU MHz:             3645.945
CPU max MHz:         3800.0000
CPU min MHz:         800.0000
BogoMIPS:            4200.00
L1d cache:           48K
L1i cache:           32K
L2 cache:            2048K
L3 cache:            107520K
NUMA node0 CPU(s):   0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94
NUMA node1 CPU(s):   1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cat_l2 cdp_l3 invpcid_single cdp_l2 ssbd mba ibrs ibpb stibp ibrs_enhanced fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect avx_vnni avx512_bf16 wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke waitpkg avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdiri movdir64b enqcmd fsrm md_clear serialize tsxldtrk pconfig arch_lbr amx_bf16 avx512_fp16 amx_tile amx_int8 flush_l1d arch_capabilities
```

```
$ nvidia-smi
Fri Feb 23 11:51:11 2024       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100 80GB HBM3          On  | 00000000:19:00.0 Off |                    0 |
| N/A   33C    P0              72W / 700W |      2MiB / 81559MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

```
jdh4@della-j11g1:~$ nvidia-smi -a
==============NVSMI LOG==============
Timestamp                                 : Fri Feb 23 11:51:29 2024
Driver Version                            : 545.23.08
CUDA Version                              : 12.3

Attached GPUs                             : 1
GPU 00000000:19:00.0
    Product Name                          : NVIDIA H100 80GB HBM3
    Product Brand                         : NVIDIA
    Product Architecture                  : Hopper
    Display Mode                          : Enabled
    Display Active                        : Disabled
    Persistence Mode                      : Enabled
    Addressing Mode                       : None
    MIG Mode
        Current                           : Disabled
        Pending                           : Disabled
    Accounting Mode                       : Disabled
    Accounting Mode Buffer Size           : 4000
    Driver Model
        Current                           : N/A
        Pending                           : N/A
    Serial Number                         : 1654123038646
    GPU UUID                              : GPU-10f35015-e921-bfab-2eb8-4e9b6664d5f1
    Minor Number                          : 0
    VBIOS Version                         : 96.00.74.00.0D
    MultiGPU Board                        : No
    Board ID                              : 0x1900
    Board Part Number                     : 692-2G520-0200-000
    GPU Part Number                       : 2330-885-A1
    FRU Part Number                       : N/A
    Module ID                             : 2
    Inforom Version
        Image Version                     : G520.0200.00.05
        OEM Object                        : 2.1
        ECC Object                        : 7.16
        Power Management Object           : N/A
    Inforom BBX Object Flush
        Latest Timestamp                  : 2024/02/22 13:09:29.459
        Latest Duration                   : 119019 us
    GPU Operation Mode
        Current                           : N/A
        Pending                           : N/A
    GSP Firmware Version                  : N/A
    GPU C2C Mode                          : Disabled
    GPU Virtualization Mode
        Virtualization Mode               : None
        Host VGPU Mode                    : N/A
    GPU Reset Status
        Reset Required                    : No
        Drain and Reset Recommended       : No
    IBMNPU
        Relaxed Ordering Mode             : N/A
    PCI
        Bus                               : 0x19
        Device                            : 0x00
        Domain                            : 0x0000
        Device Id                         : 0x233010DE
        Bus Id                            : 00000000:19:00.0
        Sub System Id                     : 0x16C110DE
        GPU Link Info
            PCIe Generation
                Max                       : 5
                Current                   : 5
                Device Current            : 5
                Device Max                : 5
                Host Max                  : 5
            Link Width
                Max                       : 16x
                Current                   : 16x
        Bridge Chip
            Type                          : N/A
            Firmware                      : N/A
        Replays Since Reset               : 0
        Replay Number Rollovers           : 0
        Tx Throughput                     : 464 KB/s
        Rx Throughput                     : 2593 KB/s
        Atomic Caps Inbound               : N/A
        Atomic Caps Outbound              : FETCHADD_32 FETCHADD_64 SWAP_32 SWAP_64 CAS_32 CAS_64 
    Fan Speed                             : N/A
    Performance State                     : P0
    Clocks Event Reasons
        Idle                              : Active
        Applications Clocks Setting       : Not Active
        SW Power Cap                      : Not Active
        HW Slowdown                       : Not Active
            HW Thermal Slowdown           : Not Active
            HW Power Brake Slowdown       : Not Active
        Sync Boost                        : Not Active
        SW Thermal Slowdown               : Not Active
        Display Clock Setting             : Not Active
    FB Memory Usage
        Total                             : 81559 MiB
        Reserved                          : 328 MiB
        Used                              : 2 MiB
        Free                              : 81227 MiB
    BAR1 Memory Usage
        Total                             : 131072 MiB
        Used                              : 1 MiB
        Free                              : 131071 MiB
    Conf Compute Protected Memory Usage
        Total                             : 0 MiB
        Used                              : 0 MiB
        Free                              : 0 MiB
    Compute Mode                          : Default
    Utilization
        Gpu                               : 0 %
        Memory                            : 0 %
        Encoder                           : 0 %
        Decoder                           : 0 %
        JPEG                              : 0 %
        OFA                               : 0 %
    Encoder Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    FBC Stats
        Active Sessions                   : 0
        Average FPS                       : 0
        Average Latency                   : 0
    ECC Mode
        Current                           : Enabled
        Pending                           : Enabled
    ECC Errors
        Volatile
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
        Aggregate
            SRAM Correctable              : 0
            SRAM Uncorrectable            : 0
            DRAM Correctable              : 0
            DRAM Uncorrectable            : 0
    Retired Pages
        Single Bit ECC                    : N/A
        Double Bit ECC                    : N/A
        Pending Page Blacklist            : N/A
    Remapped Rows
        Correctable Error                 : 0
        Uncorrectable Error               : 0
        Pending                           : No
        Remapping Failure Occurred        : No
        Bank Remap Availability Histogram
            Max                           : 2560 bank(s)
            High                          : 0 bank(s)
            Partial                       : 0 bank(s)
            Low                           : 0 bank(s)
            None                          : 0 bank(s)
    Temperature
        GPU Current Temp                  : 33 C
        GPU T.Limit Temp                  : 54 C
        GPU Shutdown T.Limit Temp         : -8 C
        GPU Slowdown T.Limit Temp         : -2 C
        GPU Max Operating T.Limit Temp    : 0 C
        GPU Target Temperature            : N/A
        Memory Current Temp               : 41 C
        Memory Max Operating T.Limit Temp : 0 C
    GPU Power Readings
        Power Draw                        : 72.02 W
        Current Power Limit               : 700.00 W
        Requested Power Limit             : 700.00 W
        Default Power Limit               : 700.00 W
        Min Power Limit                   : 200.00 W
        Max Power Limit                   : 700.00 W
    GPU Memory Power Readings 
        Power Draw                        : 47.78 W
    Module Power Readings
        Power Draw                        : N/A
        Current Power Limit               : N/A
        Requested Power Limit             : N/A
        Default Power Limit               : N/A
        Min Power Limit                   : N/A
        Max Power Limit                   : N/A
    Clocks
        Graphics                          : 345 MHz
        SM                                : 345 MHz
        Memory                            : 2619 MHz
        Video                             : 765 MHz
    Applications Clocks
        Graphics                          : 1980 MHz
        Memory                            : 2619 MHz
    Default Applications Clocks
        Graphics                          : 1980 MHz
        Memory                            : 2619 MHz
    Deferred Clocks
        Memory                            : N/A
    Max Clocks
        Graphics                          : 1980 MHz
        SM                                : 1980 MHz
        Memory                            : 2619 MHz
        Video                             : 1545 MHz
    Max Customer Boost Clocks
        Graphics                          : 1980 MHz
    Clock Policy
        Auto Boost                        : N/A
        Auto Boost Default                : N/A
    Voltage
        Graphics                          : 670.000 mV
    Fabric
        State                             : Completed
        Status                            : Success
    Processes                             : None
```

```
$ numactl -H
available: 2 nodes (0-1)
node 0 cpus: 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94
node 0 size: 515020 MB
node 0 free: 509047 MB
node 1 cpus: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 77 79 81 83 85 87 89 91 93 95
node 1 size: 516037 MB
node 1 free: 489964 MB
node distances:
node   0   1 
  0:  10  21 
  1:  21  10 
```

```
$ nvidia-smi topo -m
	GPU0	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	PIX	PIX	SYS	SYS	SYS	SYS	0	0		N/A
NIC0	PIX	 X 	PIX	SYS	SYS	SYS	SYS				
NIC1	PIX	PIX	 X 	SYS	SYS	SYS	SYS				
NIC2	SYS	SYS	SYS	 X 	SYS	SYS	SYS				
NIC3	SYS	SYS	SYS	SYS	 X 	PIX	SYS				
NIC4	SYS	SYS	SYS	SYS	PIX	 X 	SYS				
NIC5	SYS	SYS	SYS	SYS	SYS	SYS	 X 				

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1
  NIC2: mlx5_2
  NIC3: mlx5_3
  NIC4: mlx5_4
  NIC5: mlx5_5
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

For the V100's on Adroit or Traverse would be:

```
nvcc -O3 --use_fast_math --gpu-architecture=sm_70 -o myapp myapp.cu
```

## TigerGPU

TigerGPU was composed of 80 Intel Broadwell nodes each with 4 NVIDIA P100 GPUs. See the P100 [technical specs](https://www.techpowerup.com/gpu-specs/tesla-p100-pcie-16-gb.c2888). Each GPU had 56 streaming multiprocessors (SM) and 64 CUDA FP32 cores per SM. TigerGPU has been retired.

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
| Adroit     |      1           |     4         |  A100            | 6912   | 108  | 80 |
| Adroit     |      1           |     4         |  A100            | 6912   | 108  | 40 |
| Adroit     |      1           |     4         |  V100            | 5120   | 80  | 32 |    
| Della      |      4           |     8         |  H100            | 14592  | 132 | 80 |
| Della      |     69           |     4         |  A100            | 6912   | 108  | 80 |
| Della      |     20           |     2         |  A100            | 6912   | 108  | 40 |
| Della      |     2            |    28         |  A100            | --     | --   | 10 |  
| Stella     |     6            |     2         |  A100            | 6912   | 108  | 40 |
| TigerGPU   |     80           |     4         |  P100            | 3584   | 56  | 16 |
| Traverse   |     46           |     4         |  V100            | 5120   | 80  | 32 | 

SM is streaming multiprocessor. Note that the V100 GPUs have 640 [Tensor Cores](https://devblogs.nvidia.com/cuda-9-features-revealed/) (8 per SM) where half-precision Warp Matrix-Matrix and Accumulate (WMMA) operations can be carried out. That is, each core can perform a 4x4 matrix-matrix multiply and add the result to a third matrix. There are differences between the V100 node on Adroit and the Traverse nodes (see [PCIe versus SXM2](https://www.nextplatform.com/micro-site-content/achieving-maximum-compute-throughput-pcie-vs-sxm2/)).


## GPU Hackathon at Princeton

The next hackathon will take place in [June of 2024](https://www.openhackathons.org/s/siteevent/a0C5e000008dWhxEAE/se000286). This is a great opportunity to get help from experts in porting your code to a GPU. Or you can participate as a mentor and help a team rework their code. See the [GPU Computing](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) page for details.
