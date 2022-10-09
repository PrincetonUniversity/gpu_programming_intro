from time import perf_counter
import numpy as np
import cupy as cp

N = 1000
X = cp.random.randn(N, N, dtype=np.float64)

trials = 5
times = []
for _ in range(trials):
  t0 = perf_counter()
  u, s, v = cp.linalg.decomposition.svd(X)
  cp.cuda.Device(0).synchronize()
  times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print(cp.asnumpy(s).sum())
print("CuPy version: ", cp.__version__)
