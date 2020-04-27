# GPU Utilities

This page presents common tools and utilities to aid in using GPUs.

# nvidia-smi

This is the NVIDIA Systems Management Interface. This utility can be used to monitor GPU usage and GPU memory usage. It is a comprehensive tool with many options.

```
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

`nvidia-smi` has many options. Here is an an example that produces a CSV file of various metrics:

```
$ nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

# gpustat

An alternative to `nvidia-smi` is gpustat. It also pulls its data from the NVML.

```
$ gpustat
traverse-k01g4  Sat Oct 19 11:31:51 2019
[0] Tesla V100-SXM2-32GB | 38'C,   0 % |     0 / 32480 MB |
[1] Tesla V100-SXM2-32GB | 44'C,   0 % |     0 / 32480 MB |
[2] Tesla V100-SXM2-32GB | 36'C,   0 % |     0 / 32480 MB |
[3] Tesla V100-SXM2-32GB | 44'C,   0 % |     0 / 32480 MB |
```

For a comparison of various GPU tools see [this post](https://www.andrey-melentyev.com/monitoring-gpus.html).

# nvprof

This is the NVIDIA profiler. It can be used to identify the "hot spots" in the code or the parts which are running slow and need attention. `nvprof` has a summary mode and trace mode.

### Summary mode

**Step 1**: Run your code under nvprof:

```
#!/bin/bash
#SBATCH --job-name=cuda_c        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3 cudatoolkit

srun nvprof <executable> <inputfile1>
```

**Step 2**: View the Slurm output for percentage of time and total time for GPU (top of output) and CPU (bottom of output, i.e., calls to C API).

```
$ cat slurm-670855.out
==25861== NVPROF is profiling process 25861, command: python pt_svd.py
Execution time:  14.978624082170427
Result:  214726.22269399266
PyTorch version:  1.3.0
==25861== Profiling application: python pt_svd.py
==25861== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   32.17%  1.23343s      2976  414.46us  60.289us  1.0076ms  void gemv2N_kernel<int, int, double, double, double, int=128, int=4, int=4, int=4, int=11, cublasGemvParams<cublasGemvTensor<double const >, cublasGemvTensor<double>, double>>(double const )
                   30.12%  1.15485s      3872  298.26us  4.8640us  888.84us  void gemv2T_kernel_val<int, int, double, double, double, int=128, int=16, int=2, int=2, bool=0, cublasGemvParams<cublasGemvTensor<double const >, cublasGemvTensor<double>, double>>(double const , double, double)
                   16.12%  618.00ms      8123  76.080us  1.0240us  98.512ms  [CUDA memcpy HtoD]
                    6.64%  254.46ms      7989  31.851us  1.6000us  103.47ms  [CUDA memcpy DtoH]
                    6.18%  237.03ms       171  1.3861ms  66.401us  4.5433ms  dgemm_sm_heavy_ldg_nt
                    4.35%  166.62ms       140  1.1902ms  63.489us  4.4663ms  dgemm_sm_heavy_ldg_nn
                    1.84%  70.391ms        32  2.1997ms  68.225us  4.3317ms  dgemm_sm35_ldg_tn_64x8x128x8x32
                    0.63%  23.998ms        63  380.92us  35.936us  398.15us  void trmm_right_kernel_core<double, int=256, int=4, int=128, bool=0, bool=1, bool=0, bool=1, bool=1>(cublasTrmmParams<double>, double, int)
                    0.60%  23.165ms       156  148.49us  22.976us  860.68us  dgemm_sm35_ldg_nt_64x8x128x8x32
                    0.60%  22.829ms       512  44.587us  18.145us  82.401us  void gemv2N_kernel<int, int, double, double, double, int=128, int=8, int=4, int=4, int=11, cublasGemvParams<cublasGemvTensor<double const >, cublasGemvTensor<double>, double>>(double const )
                    0.21%  7.9724ms       124  64.293us  23.361us  146.85us  dgemm_sm35_ldg_nn_64x8x128x8x32
                    0.20%  7.7708ms       113  68.768us  28.001us  136.51us  dgemm_sm35_ldg_nn_128x8x64x16x16
                    0.18%  6.7496ms       113  59.731us  31.584us  119.49us  dgemm_sm35_ldg_nt_128x8x64x16x16
                    0.12%  4.4888ms       384  11.689us  4.8000us  21.408us  void gemvNSP_kernel<double, double, double, int=11, int=32, int=4, int=1024, cublasGemvParams<cublasGemvTensor<double const >, cublasGemvTensor<double>, double>>(double const )
                    0.05%  2.0084ms         1  2.0084ms  2.0084ms  2.0084ms  _ZN84_GLOBAL__N__60_tmpxft_00007dcf_00000000_11_Distributions_compute_75_cpp1_ii_c3aa7ee643distribution_elementwise_grid_stride_kernelIdLi2EZZZN2at6native18normal_kernel_cudaERNS1_14TensorIteratorEddPNS1_9GeneratorEENKUlvE_clEvENKUlvE_clEvEUlP24curandStatePhilox4_32_10E_ZNS_27distribution_nullary_kernelIddLi2ESB_ZZZNS2_18normal_kernel_cudaES4_ddS6_ENKS7_clEvENKS8_clEvEUldE_EEvS4_PNS1_13CUDAGeneratorERKT2_T3_EUlidE_EEviSt4pairImmET1_SG_
                    0.00%  37.697us         1  37.697us  37.697us  37.697us  void trmm_right_kernel_core<double, int=256, int=4, int=128, bool=0, bool=0, bool=0, bool=1, bool=1>(cublasTrmmParams<double>, double, int)
                    0.00%  10.912us         1  10.912us  10.912us  10.912us  _ZN2at6native13reduce_kernelILi512ENS0_8ReduceOpIdNS0_14func_wrapper_tIdZNS0_15sum_kernel_implIdddEEvRNS_14TensorIteratorEEUlddE_EEjdLi4EEEEEvT0_
      API calls:   30.19%  3.01664s        25  120.67ms  5.2080us  2.99951s  cudaMalloc
                   29.24%  2.92161s        30  97.387ms     519ns  2.06221s  cudaFree
                   28.38%  2.83543s     16109  176.02us  2.8410us  17.324ms  cudaStreamSynchronize
                    5.42%  541.66ms      3947  137.23us  4.6810us  104.42ms  cudaMemcpyAsync
                    4.60%  459.22ms         4  114.81ms  1.1590us  397.85ms  cudaHostAlloc
                    1.19%  118.47ms     12162  9.7400us  3.7080us  8.1657ms  cudaMemcpy2DAsync
                    0.88%  88.234ms      8659  10.189us  5.1420us  1.3428ms  cudaLaunchKernel
                    0.02%  2.1831ms       379  5.7600us     109ns  228.06us  cuDeviceGetAttribute
                    0.02%  2.0618ms      8244     250ns     109ns  63.772us  cudaGetLastError
                    0.02%  1.6867ms         4  421.68us  412.95us  438.45us  cuDeviceTotalMem
                    0.01%  1.1086ms         2  554.28us  551.40us  557.16us  cudaGetDeviceProperties
                    0.01%  947.66us       383  2.4740us  2.2970us  15.374us  cudaFuncGetAttributes
                    0.01%  659.29us       958     688ns     401ns  9.9460us  cudaStreamWaitEvent
                    0.01%  518.28us       734     706ns     378ns  5.3830us  cudaEventRecord
                    0.00%  274.21us         4  68.552us  50.274us  100.64us  cuDeviceGetName
                    0.00%  232.29us         3  77.431us  57.122us  110.35us  cudaStreamCreate
                    0.00%  180.44us         3  60.148us  15.615us  97.570us  cudaMemcpy
                    0.00%  149.91us         6  24.985us  2.8710us  67.828us  cudaStreamCreateWithFlags
                    0.00%  139.64us       255     547ns     309ns  4.5420us  cudaStreamGetPriority
                    0.00%  108.72us        74  1.4690us     241ns  26.083us  cudaGetDevice
                    0.00%  102.40us        31  3.3030us     280ns  34.893us  cudaSetDevice
                    0.00%  72.471us         9  8.0520us  2.9420us  21.052us  cudaStreamDestroy
                    0.00%  66.483us        54  1.2310us     368ns  8.1740us  cudaEventCreateWithFlags
                    0.00%  44.999us        54     833ns     342ns  3.8350us  cudaEventDestroy
                    0.00%  37.294us         9  4.1430us  1.6000us  7.6300us  cudaDeviceSynchronize
                    0.00%  24.171us        63     383ns     233ns  1.7160us  cudaDeviceGetAttribute
                    0.00%  5.9150us        20     295ns      90ns  2.7330us  cudaGetDeviceCount
                    0.00%  4.2950us         1  4.2950us  4.2950us  4.2950us  cuDeviceGetPCIBusId
                    0.00%  3.8810us         4     970ns     234ns  2.1650us  cuDevicePrimaryCtxGetState
                    0.00%  3.2070us         3  1.0690us     908ns  1.1590us  cuInit
                    0.00%  2.5280us         6     421ns     151ns     769ns  cuDeviceGetCount
                    0.00%  2.0280us         5     405ns     131ns     550ns  cuDeviceGet
                    0.00%  1.7500us         3     583ns     490ns     655ns  cuDriverGetVersion
                    0.00%     804ns         4     201ns     199ns     207ns  cuDeviceGetUuid
```

### Trace mode

**Step 1**: Run your code under nvprof:

```
#!/bin/bash
#SBATCH --job-name=cuda_c        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3 cudatoolkit

srun nvprof -o trace.sqlite --print-gpu-trace <executable> <inputfile1> 
```

**Step 2**: Use the [NVIDIA visual profiler](https://developer.nvidia.com/nvidia-visual-profiler) to view the output:

Note that you may need to do this on your own laptop by first installing the CUDA toolkit. It is presently not working on Adroit.

```
# X11 forwarding must be enabled for the next line (i.e., ssh -X)
$ nvvp trace.sqlite
```

See the documentation [here](https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvprof-overview).

The following lines can be used to profile CuPy code:

```python
from cupy.cuda.nvtx import RangePush
from cupy.cuda.nvtx import RangePop
RangePush('Query score', 0)
# do work here with CuPy
RangePop()
```

Then do `nvvp` on the output of the trace file.

# nvcc

This is the NVIDIA CUDA compiler. It is based on LLVM. To compile a simple code:

```
$ nvcc -o hello_world hello_world.cu
```

# TigerGPU Utilization Dashboard

See [this page](https://researchcomputing.princeton.edu/node/7171) to view the GPU usage on TigerGPU.

# ARM DDT

The general directions for using the DDT debugger are [here](https://researchcomputing.princeton.edu/faq/debugging-with-ddt-on-the).

```
$ ssh -X <NetID>@adroit.princeton.edu
$ git clone https://github.com/PrincetonUniversity/hpc_beginning_workshop
$ cd hpc_beginning_workshop/RC_example_jobs/simple_gpu_kernel
$ salloc -N 1 -n 1 -t 10:00 --gres=gpu:tesla_k40c:1 --x11
$ module load cudatoolkit/10.1
$ nvcc -g -G hello_world_gpu.cu
$ module load ddt/20.0.1
$ export ALLINEA_FORCE_CUDA_VERSION=10.1
$ ddt
# check cuda, uncheck "submit to queue", and click on "Run"
```

The `-g` debugging flag is for CPU code while the `-G` flag is for GPU code. `-G` turns off compiler optimizations. Note that as of February 2020 CUDA Toolkit 10.2 is not supported.

If the graphics are not displaying fast enough then consider using [TurboVNC](https://researchcomputing.princeton.edu/faq/how-do-i-use-vnc-on-tigre).
