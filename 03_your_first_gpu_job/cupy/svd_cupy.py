from time import perf_counter

N = 1000
cpu_runs = 10
gpu_runs = 10

times = []
import numpy as np
print("NumPy version: ", np.__version__)
X = np.random.randn(N, N).astype(np.float32)
for _ in range(cpu_runs):
  t0 = perf_counter()
  np.linalg.svd(X)
  times.append(perf_counter() - t0)
print("CPU time: ", min(times))

times = []
import cupy as cp
print("CuPy version: ", cp.__version__)
X = cp.random.randn(N, N, dtype=np.float32)
for _ in range(gpu_runs):
  t0 = perf_counter()
  cp.linalg.decomposition.svd(X)
  times.append(perf_counter() - t0)
print("GPU time: ", min(times))
