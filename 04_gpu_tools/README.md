# GPU Tools

This page presents common tools and utilities for GPU computing.

# nvidia-smi

This is the NVIDIA Systems Management Interface. This utility can be used to monitor GPU usage and GPU memory usage. It is a comprehensive tool with many options.

```
$ nvidia-smi
Wed Jul  9 14:13:17 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100 80GB PCIe          On  |   00000000:4A:00.0 Off |                    0 |
| N/A   42C    P0             97W /  300W |    1187MiB /  81920MiB |     32%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         2308208      C   python                                 1178MiB |
+-----------------------------------------------------------------------------------------+
```

To see all of the available options, view the help:

```$ nvidia-smi --help```

Here is an an example that produces CSV output of various metrics:

```
$ nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5
```

The command above takes a reading every 5 seconds.

# Nsight Systems (nsys) for Profiling

The `nsys` command can be used to generate a timeline of the execution of your code. `nsys-ui` provides a GUI to examine the profiling data generated by `nsys`. See the NVIDIA Nsight Systems [getting started guide](https://docs.nvidia.com/nsight-systems/) and notes on [Summit](https://docs.olcf.ornl.gov/systems/summit_user_guide.html#profiling-gpu-code-with-nvidia-developer-tools).

To see the help menu:

```
$ /usr/local/bin/nsys --help
$ /usr/local/bin/nsys --help profile
```

IMPORTANT: Do not run profiling jobs in your `/home` directory because large files are often written during these jobs which can exceed your quota. Instead launch jobs from `/scratch/gpfs/<YourNetID>` where you have lots of space. Here's an example:

```
$ ssh <YourNetID>@della-gpu.princeton.edu
$ cd /scratch/gpfs/<YourNetID>
$ mkdir myjob && cd myjob
# prepare Slurm script
$ sbatch job.slurm
```

Below is an example Slurm script:

```
#!/bin/bash
#SBATCH --job-name=profile       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.10
conda activate myenv

/usr/local/bin/nsys profile --trace=cuda,nvtx,osrt -o myprofile_${SLURM_JOBID} python myscript.py
```

For an MPI code you should use:

```
srun --wait=0 /usr/local/bin/nsys profile --trace=cuda,nvtx,osrt,mpi -o myprofile_${SLURM_JOBID} ./my_mpi_exe
```

Run this command to see the summary statistics:

```
$ /usr/local/bin/nsys stats myprofile_*.nsys-rep
```

To work the the graphical interface (nsys-ui) you can either (1) download the `.qdrep` file to your local machine or (2) create a graphical desktop session on [https://mydella.princeton.edu](https://mydella.princeton.edu/) or [https://mystellar.princeton.edu](https://mystellar.princeton.edu/). To create the graphical desktop, choose "Interactive Apps" then "Desktop of Della/Stellar Vis Nodes". Once the session starts, click on the black terminal icon and then run:

```
$ /usr/local/bin/nsys-ui myprofile_*.nsys-rep
```

# Nsight Compute (ncu) for GPU Kernel Profiling

The `ncu` command is used for detailed profiling of GPU kernels. See the NVIDIA [documentation](https://docs.nvidia.com/nsight-compute/). On some clusters you will need to load a module to make the command available:

```
$ module load cudatoolkit/12.9
$ ncu --help
```

The idea is to use `ncu` for the profiling and `ncu-ui` for examining the data in a GUI.

Below is a sample slurm script:

```
#!/bin/bash
#SBATCH --job-name=profile       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
conda activate myenv

ncu -o my_report_${SLURM_JOBID} python myscript.py
```

Note: the `ncu` profiler can significantly slow down the execution time of the code.

To work the the graphical interface (ncu-ui) you can either (1) download the `.ncu-rep` file to your local machine or (2) create a graphical desktop session on [https://mydella.princeton.edu](https://mydella.princeton.edu/) or [https://mystellar.princeton.edu](https://mystellar.princeton.edu/). To create the graphical desktop, choose "Interactive Apps" then "Desktop of Della/Stellar Vis Nodes". Once the session starts, click on the black terminal icon and then run:

```
$ module load cudatoolkit/12.9
$ ncu-ui my_report_*.ncu-rep
```

# line_prof for Python Profiling

The [line_prof](https://researchcomputing.princeton.edu/python-profiling) tool provides profiling info for each line of a function. It is easy to use and it can be used for Python codes that run on CPUs and/or GPUs.

# nvcc

This is the NVIDIA CUDA compiler. It is based on LLVM. To compile a simple code:

```
$ module load cudatoolkit/12.9
$ nvcc -o hello_world hello_world.cu
```

# Job Statistics

Follow [this procedure](https://researchcomputing.princeton.edu/support/knowledge-base/jobstats) to view detailed metrics for your Slurm jobs. This includes GPU utilization and memory as a function of time.

# GPU Computing

See [this page](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing) for an overview of the hardware at Princton as well as useful commands like `gpudash` and `shownodes`.

# Debuggers

### ARM DDT

The general directions for using the DDT debugger are [here](https://researchcomputing.princeton.edu/faq/debugging-with-ddt-on-the). The getting started guide is [here](https://developer.arm.com/tools-and-software/server-and-hpc/debug-and-profile/arm-forge/arm-ddt).

```
$ ssh -X <NetID>@adroit.princeton.edu  # better to use graphical desktop via myadroit
$ git clone https://github.com/PrincetonUniversity/hpc_beginning_workshop
$ cd hpc_beginning_workshop/RC_example_jobs/simple_gpu_kernel
$ salloc -N 1 -n 1 -t 10:00 --gres=gpu:1 --x11
$ module load cudatoolkit/12.9
$ nvcc -g -G hello_world_gpu.cu
$ module load ddt/24.1
$ #export ALLINEA_FORCE_CUDA_VERSION=10.1
$ ddt
# check cuda, uncheck "submit to queue", and click on "Run"
```

The `-g` debugging flag is for CPU code while the `-G` flag is for GPU code. `-G` turns off compiler optimizations.

If the graphics are not displaying fast enough then consider using [TurboVNC](https://researchcomputing.princeton.edu/faq/how-do-i-use-vnc-on-tigre).

### `cuda-gdb`

`cuda-gdb` is a free debugger available as part of the CUDA Toolkit.
