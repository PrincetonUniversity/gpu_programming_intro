# Your First GPU Job

Using the GPUs on the Princeton HPC clusters is easy. Pick one of the applications below to get started. To obtain the materials to run the examples, use these commands:

```
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/network/<YourNetID>
$ git clone https://github.com/PrincetonUniversity/gpu_programming_intro.git
```

## CuPy

[CuPy](https://cupy.chainer.org) provides a Python interface to set of common numerical routines (e.g., matrix factorizations) which are executed on a GPU (see the [Reference Manual](https://docs-cupy.chainer.org/en/stable/reference/index.html)). You can roughly think of CuPy as NumPy for GPUs. This example is set to use the CuPy installation of the workshop instructor. If you use CuPy for your research work then you should [install it](https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/02_cuda_toolkit#conda-installations) into your account.

Examine the Python script before running the code:

```python
$ cd gpu_programming_intro/03_your_first_gpu_job/cupy
$ cat svd.py
from time import perf_counter
import numpy as np
import cupy as cp

N = 1000
X = cp.random.randn(N, N, dtype=np.float64)

trials = 5
times = []
for _ in range(trials):
  t0 = perf_counter()
  u, s, v = cp.linalg.svd(X)
  cp.cuda.Device(0).synchronize()
  times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print(cp.asnumpy(s).sum())
print("CuPy version: ", cp.__version__)
```

Below is a sample Slurm script:

```bash
$ cat job.slurm
#!/bin/bash
#SBATCH --job-name=cupy-job      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2022.5
conda activate /scratch/network/jdh4/.gpu_workshop/envs/cupy-env

python svd.py
```

A GPU is allocated using the Slurm directive `#SBATCH --gres=gpu:1`.

Submit the job:

```
$ sbatch job.slurm
```

You can monitor the progress of the job with `squeue -u $USER`. Once the job completes, view the output with `cat slurm-*.out`. What happens if you re-run the script with the matrix in single precision? Does the execution time double if N is doubled? There is a CPU version of the code at the bottom of this page. Does the operation run faster on the CPU with NumPy or on the GPU with CuPy? Try [this exercise](https://github.com/PrincetonUniversity/a100_workshop/tree/main/06_cupy#cupy-uses-tensor-cores) where the Tensor Cores are utilized by using less than single precision (i.e., TensorFloat32).

Why are multiple trials used when measuring the execution time? `CuPy` compiles a custom GPU kernel for each GPU operation (e.g., SVD). This means the first time a `CuPy` function is called the measured time is the sum of the compile time plus the time to execute the operation. The second and later calls only include the time to execute the operation.

In addition to CuPy, Python programmers looking to run their code on GPUs should also be aware of [Numba](https://numba.pydata.org/) and [JAX](https://github.com/google/jax).

## PyTorch

[PyTorch](https://pytorch.org) is a popular deep learning framework. See its documentation for [Tensor operations](https://pytorch.org/docs/stable/tensors.html). This example is set to use the PyTorch installation of the workshop instructor. If you use PyTorch for your research work then you should [install it](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) into your account.

Examine the Python script before running the code:

```python
$ cd gpu_programming_intro/03_your_first_gpu_job/pytorch
$ cat svd.py
from time import perf_counter
import torch

N = 1000

cuda0 = torch.device('cuda:0')
x = torch.randn(N, N, dtype=torch.float64, device=cuda0)
t0 = perf_counter()
u, s, v = torch.svd(x)
elapsed_time = perf_counter() - t0

print("Execution time: ", elapsed_time)
print("Result: ", torch.sum(s).cpu().numpy())
print("PyTorch version: ", torch.__version__)
```

Here is a sample Slurm script:

```bash
$ cat job.slurm
#!/bin/bash
#SBATCH --job-name=torch-svd     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # choose a100 or v100 on adroit
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module load anaconda3/2022.5
conda activate /scratch/network/jdh4/.gpu_workshop/envs/torch-env

python svd.py
```

Submit the job:

```
$ sbatch job.slurm
```

You can monitor the progress of the job with `squeue -u $USER`. Once the job completes, view the output with `cat slurm-*.out`.

## TensorFlow

[TensorFlow](https://www.tensorflow.org) is popular library for training deep neural networks. It can also be used for various numerical computations (see [documentation](https://www.tensorflow.org/api_docs/python/tf)). This example is set to use the TensorFlow installation of the workshop instructor. If you use TensorFlow for your research work then you should [install it](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow) into your account.

Examine the Python script before running the code:

```python
$ cd gpu_programming_intro/03_your_first_gpu_job/tensorflow
$ cat svd.py
from time import perf_counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

N = 100
x = tf.random.normal((N, N), dtype=tf.dtypes.float64)
t0 = perf_counter()
s, u, v = tf.linalg.svd(x)
elapsed_time = perf_counter() - t0
print("Execution time: ", elapsed_time)
print("Result: ", tf.reduce_sum(s).numpy())
```

Below is a sample Slurm script:

```bash
$ cat job.slurm
#!/bin/bash
#SBATCH --job-name=svd-tf        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module load anaconda3/2022.5
conda activate /scratch/network/jdh4/.gpu_workshop/envs/tf-gpu

python svd.py
```

Submit the job:

```
$ sbatch job.slurm
```

You can monitor the progress of the job with `squeue -u $USER`. Once the job completes, view the output with `cat slurm-*.out`.

<!--### Benchmarks

Below is benchmark data for the SVD of an N x N matrix in double precision using NumPy with a single CPU-core on Adroit versus TensorFlow on Traverse using a single CPU-core and a V100 GPU:

![svd-data](svd_adroit_traverse_log_log.png)-->

## R with NVBLAS

Take a look at [this page](https://github.com/PrincetonUniversity/HPC_R_Workshop/tree/master/07_NVBLAS) and then run the commands below:

```
$ git clone https://github.com/PrincetonUniversity/HPC_R_Workshop
$ cd HPC_R_Workshop/07_NVBLAS
$ mv nvblas.conf ~
$ sbatch 07_NVBLAS.cmd
```

Here is the sample output:

```
$ cat slurm-*.out
...
[1] "Matrix multiply:"
   user  system elapsed 
  0.166   0.137   0.304 
[1] "----"
[1] "Cholesky Factorization:"
   user  system elapsed 
  1.053   0.041   1.096 
[1] "----"
[1] "Singular Value Decomposition:"
   user  system elapsed 
  8.060   1.837   5.345 
[1] "----"
[1] "Principal Components Analysis:"
   user  system elapsed 
 16.814   5.987  11.252 
[1] "----"
[1] "Linear Discriminant Analysis:"
   user  system elapsed 
 25.955   3.080  20.830 
[1] "----"
...
```

See the [user guide](https://docs.nvidia.com/cuda/nvblas/index.html) for NVBLAS.

## MATLAB

MATLAB is already installed on the cluster. Simply follow these steps:

```bash
$ cd gpu_programming_intro/03_your_first_gpu_job/matlab
$ cat svd.m
```

Here is the MATLAB script:

```matlab
gpu = gpuDevice();
fprintf('Using a %s GPU.\n', gpu.Name);
disp(gpuDevice);

X = gpuArray([1 0 2; -1 5 0; 0 3 -9]);
whos X
[U,S,V] = svd(X)
fprintf('trace(S): %f\n', trace(S))
quit;
```

Below is a sample Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=matlab-svd    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load matlab/R2022a

matlab -singleCompThread -nodisplay -nosplash -r svd
```

Submit the job:

```
$ sbatch job.slurm
```

You can monitor the progress of the job with `squeue -u $USER`. Once the job completes, view the output with `cat slurm-*.out`. Learn more about [MATLAB on the Research Computing clusters](https://researchcomputing.princeton.edu/support/knowledge-base/matlab).

Here is an [intro](https://www.mathworks.com/help/parallel-computing/run-matlab-functions-on-a-gpu.html) to using MATLAB with GPUs.

## Julia

See the section on "Julia Environments and GPU Packages" on [this page](https://researchcomputing.princeton.edu/support/knowledge-base/julia).

## Monitoring GPU Usage

To monitor jobs in our reservation:

```
$ watch -n 1 squeue -R <reservation-name>
```

## Benchmarks

### Matrix Multiplication

| cluster              | code |  CPU-cores  | time (s) |
|:--------------------:|:----:|:-----------:|:--------:|
|  adroit (CPU)        | NumPy |    1       |  24.2    |
|  adroit (CPU)        | NumPy |    2       |  15.5    |
|  adroit (CPU)        | NumPy |    4       |   5.3    |  
|  adroit (V100)       | CuPy  |    1       |   0.3   |
|  adroit (K40c)       | CuPy  |    1       |   1.7   |

Times are best of 5 for a square matrix with N=10000 in double precision.

### LU Decomposition

| cluster              | code        |  CPU-cores | time (s) |
|:--------------------:|:-----------:|:----------:|:--------:|
|  adroit (CPU)        | SciPy       |    1       |   9.4   |
|  adroit (CPU)        | SciPy       |    2       |   7.9   |
|  adroit (CPU)        | SciPy       |    4       |   6.5   |  
|  adroit (V100)       | CuPy        |    1       |   0.3   |
|  adroit (K40c)       | CuPy        |    1       |   1.1   |
|  adroit (V100)       | Tensorflow  |    1       |   0.3   |
|  adroit (K40c)       | Tensorflow  |    1       |   1.1   |
|  adroit (CPU)        | Tensorflow  |    1       |  50.8   |

Times are best of 5 for a square matrix with N=10000 in double precision.

### Singular Value Decomposition

| cluster              | code       |  CPU-cores | time (s) |
|:--------------------:|:----------:|:----------:|:--------:|
|  adroit (CPU)        | NumPy      |    1       |    3.6   |
|  adroit (CPU)        | NumPy      |    2       |    3.0   |
|  adroit (CPU)        | NumPy      |    4       |    1.2   |
|  adroit (V100)       | CuPy       |    1       |   24.7   |
|  adroit (K40c)       | CuPy       |    1       |   30.5   |
|  adroit (V100)       | Torch      |    1       |   0.9    |
|  adroit (K40c)       | Torch      |    1       |   1.5    |
|  adroit (CPU)        | Torch      |    1       |   3.0    |
|  adroit (V100)       | TensorFlow |    1       |   24.8   |
|  adroit (K40c)       | TensorFlow |    1       |   29.7   |
|  adroit (CPU)        | TensorFlow |    1       |    9.2   |

Times are best of 5 for a square matrix with N=2000 in double precision.

For the LU decomposition using SciPy:

```
from time import perf_counter

import numpy as np
import scipy as sp
from scipy.linalg import lu

N = 10000
cpu_runs = 5

times = []
X = np.random.randn(N, N).astype(np.float64)
for _ in range(cpu_runs):
  t0 = perf_counter()
  p, l, u = lu(X, check_finite=False)
  times.append(perf_counter() - t0)
print("CPU time: ", min(times))
print("NumPy version: ", np.__version__)
print("SciPy version: ", sp.__version__)
print(p.sum())
print(times)
```

For the LU decomposition on the CPU:

```
from time import perf_counter

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
print("TensorFlow version: ", tf.__version__)

times = []
N = 10000
with tf.device("/cpu:0"):
  x = tf.random.normal((N, N), dtype=tf.dtypes.float64)
  for _ in range(5):
    t0 = perf_counter()
    lu, p = tf.linalg.lu(x)
    elapsed_time = perf_counter() - t0
    times.append(elapsed_time)
print("Execution time: ", min(times))
print(times)
print("Result: ", tf.reduce_sum(p).numpy())
```

SVD with NumPy:

```
from time import perf_counter

N = 2000
cpu_runs = 5

times = []
import numpy as np
X = np.random.randn(N, N).astype(np.float64)
for _ in range(cpu_runs):
  t0 = perf_counter()
  u, s, v = np.linalg.svd(X)
  times.append(perf_counter() - t0)
print("CPU time: ", min(times))
print("NumPy version: ", np.__version__)
print(s.sum())
print(times)
```

Performing benchmarks with R:

```
# install.packages("microbenchmark")
library(microbenchmark)
library(Matrix)

N <- 10000
microbenchmark(lu(matrix(rnorm(N*N), N, N)), times=5, unit="s")
```
