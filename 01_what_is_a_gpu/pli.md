# PLI Nodes

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

## Intra-Node Topology

```
jdh4@della-k17g3:~$ nvidia-smi topo -m
	GPU0	GPU1	GPU2	GPU3	GPU4	GPU5	GPU6	GPU7	NIC0	NIC1	NIC2	NIC3	NIC4	NIC5	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NV18	PIX	PIX	NODE	NODE	NODE	NODE		0		N/A
GPU1	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NV18	NODE	NODE	NODE	NODE	NODE	NODE		0		N/A
GPU2	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NV18	NODE	NODE	NODE	NODE	NODE	NODE		0		N/A
GPU3	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NV18	NODE	NODE	PIX	NODE	NODE	NODE		0		N/A
GPU4	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NV18	NODE	NODE	NODE	PIX	PIX	NODE	1	1		N/A
GPU5	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NV18	NODE	NODE	NODE	NODE	NODE	NODE	1	1		N/A
GPU6	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NV18	NODE	NODE	NODE	NODE	NODE	PIX	1	1		N/A
GPU7	NV18	NV18	NV18	NV18	NV18	NV18	NV18	 X 	NODE	NODE	NODE	NODE	NODE	NODE	1	1		N/A
NIC0	PIX	NODE	NODE	NODE	NODE	NODE	NODE	NODE	 X 	PIX	NODE	NODE	NODE	NODE				
NIC1	PIX	NODE	NODE	NODE	NODE	NODE	NODE	NODE	PIX	 X 	NODE	NODE	NODE	NODE				
NIC2	NODE	NODE	NODE	PIX	NODE	NODE	NODE	NODE	NODE	NODE	 X 	NODE	NODE	NODE				
NIC3	NODE	NODE	NODE	NODE	PIX	NODE	NODE	NODE	NODE	NODE	NODE	 X 	PIX	NODE				
NIC4	NODE	NODE	NODE	NODE	PIX	NODE	NODE	NODE	NODE	NODE	NODE	PIX	 X 	NODE				
NIC5	NODE	NODE	NODE	NODE	NODE	NODE	PIX	NODE	NODE	NODE	NODE	NODE	NODE	 X 				

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
