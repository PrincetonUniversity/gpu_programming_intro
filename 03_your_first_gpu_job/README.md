# Your First GPU Job

Using the GPUs on the Princeton HPC clusters is easy. Pick one of the applications below to get started. To obtain the materials to run the examples, use these commands:

```
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/network/<YourNetID>
$ git clone https://github.com/PrincetonUniversity/gpu_programming_intro.git
```

To add a GPU to your Slurm allocation:

```
#SBATCH --gres=gpu:1             # number of gpus per node
```

For Adroit, one can specify the GPU type using a constraint:

```
#SBATCH --constraint=a100        # set to gpu80, a100 or a40
#SBATCH --gres=gpu:1             # number of gpus per node
```

For more on specifying the GPU type on Adroit [see this page](https://researchcomputing.princeton.edu/systems/adroit#gpus).

## CuPy

[CuPy](https://cupy.chainer.org) provides a Python interface to set of common numerical routines (e.g., matrix factorizations) which are executed on a GPU (see the [Reference Manual](https://docs-cupy.chainer.org/en/stable/reference/index.html)). You can roughly think of CuPy as NumPy for GPUs. This example is set to use the CuPy installation of the workshop instructor. If you use CuPy for your research work then you should [install it](https://github.com/PrincetonUniversity/gpu_programming_intro/tree/master/02_cuda_toolkit#conda-installations) into your account.

Examine the Python script before running the code:

```python
$ cd gpu_programming_intro/03_your_first_gpu_job/cupy
$ cat svd.py
from time import perf_counter
import cupy as cp

N = 1000
X = cp.random.randn(N, N, dtype=cp.float64)

trials = 5
times = []
for _ in range(trials):
    t0 = perf_counter()
    u, s, v = cp.linalg.svd(X)
    cp.cuda.Device(0).synchronize()
    times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print("sum(s) = ", s.sum())
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
module load anaconda3/2026.7
conda activate /scratch/network/jdh4/.gpu_workshop/envs/cupy-env

python svd.py
```

A GPU is allocated using the Slurm directive `#SBATCH --gres=gpu:1`.

Submit the job:

```
$ sbatch job.slurm
```

Wait a few seconds for the job to run. Inspect the output:

```
$ cat slurm-*.out
```

You can monitor the progress of the job with `squeue --me`. Once the job completes, view the output with `cat slurm-*.out`. What happens if you re-run the script with the matrix in single precision? Does the execution time double if N is doubled? Try [this exercise](https://github.com/PrincetonUniversity/a100_workshop/tree/main/06_cupy#cupy-uses-tensor-cores) where the Tensor Cores are utilized by using less than single precision (i.e., TensorFloat32).

Why are multiple trials used when measuring the execution time? `CuPy` compiles a custom GPU kernel for each GPU operation (e.g., SVD). This means the first time a `CuPy` function is called the measured time is the sum of the compile time plus the time to execute the operation. The second and later calls only include the time to execute the operation.

In addition to CuPy, Python programmers looking to run their code on GPUs should also be aware of [Numba](https://numba.pydata.org/) and [JAX](https://github.com/google/jax).

To see a performance comparison between the CPU and GPU, see `matmul_numpy.py` and `matmul_cupy.py` in [this repo](https://github.com/jdh4/python-gpu/tree/main/cupy).

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

module purge
module load anaconda3/2026.7
conda activate /scratch/network/jdh4/.gpu_workshop/envs/torch-env

python svd.py
```

Submit the job:

```
$ sbatch job.slurm
```

Wait a few seconds for the job to run. Inspect the output:

```
$ cat slurm-*.out
```

You can monitor the progress of the job with `squeue --me`. Once the job completes, view the output with `cat slurm-*.out`.

## JAX

[JAX](https://docs.jax.dev/en/latest/index.html) is popular machine learning library. It is used by Google to train Gemini. This example is set to use the JAX installation of the workshop instructor. If you use JAX for your research work then you should [install it](https://researchcomputing.princeton.edu/support/knowledge-base/jax) into your account.

Examine the Python script before running the code:

```python
$ cd gpu_programming_intro/03_your_first_gpu_job/jax
$ cat svd.py
from time import perf_counter
import jax
import jax.numpy as jnp

print("JAX version: ", jax.__version__)

N = 4000
key = jax.random.PRNGKey(42)
A = jax.random.normal(key, (N, N))
t0 = perf_counter()
U, S, Vt = jnp.linalg.svd(A, full_matrices=False)
elapsed_time = perf_counter() - t0

print("Execution time: ", elapsed_time)
print("Result: ", jnp.sum(s))
```

Below is a sample Slurm script:

```bash
$ cat job.slurm
#!/bin/bash
#SBATCH --job-name=svd-jax       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory (RAM) per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:00:30          # total run time limit (HH:MM:SS)
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module load anaconda3/2026.7
conda activate /scratch/network/jdh4/.gpu_workshop/envs/jax-gpu

python svd.py
```

Submit the job:

```
$ sbatch job.slurm
```

Wait a few seconds for the job to run. Inspect the output:

```
$ cat slurm-*.out
```

You can monitor the progress of the job with `squeue --me`. Once the job completes, view the output with `cat slurm-*.out`.

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
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --constraint=a100        # choose a100 or v100
#SBATCH --reservation=gpuprimer  # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load matlab/R2025b

matlab -singleCompThread -nodisplay -nosplash -r svd
```

Submit the job:

```
$ sbatch job.slurm
```

Wait a few seconds for the job to run. Inspect the output:

```
$ cat slurm-*.out
```

You can monitor the progress of the job with `squeue --me`. Once the job completes, view the output with `cat slurm-*.out`. Learn more about [MATLAB on the Research Computing clusters](https://researchcomputing.princeton.edu/support/knowledge-base/matlab).

Here is an [intro](https://www.mathworks.com/help/parallel-computing/run-matlab-functions-on-a-gpu.html) to using MATLAB with GPUs.

## Julia

Install the `CUDA` and `LinearAlgebra` packages then run the script in `03_your_first_gpu_job/julia`. See our [Julia webage](https://researchcomputing.princeton.edu/support/knowledge-base/julia).

## Monitoring GPU Usage

To monitor jobs in our reservation:

```
$ watch -n 1 squeue -R gpuprimer
```
