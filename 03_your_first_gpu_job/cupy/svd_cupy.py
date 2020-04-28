from time import perf_counter
import numpy as np
import cupy as cp
# import cupyx.scipy.linalg

trials = 5
times = []

N = 1000
X = cp.random.randn(N, N, dtype=np.float64)
for _ in range(trials):
  t0 = perf_counter()
  u, s, v = cp.linalg.decomposition.svd(X)
  # Y = cp.matmul(X, X)
  # lu, piv = cupyx.scipy.linalg.lu_factor(X, check_finite=False)
  cp.cuda.Device(0).synchronize()
  times.append(perf_counter() - t0)
print("Execution time: ", elapsed_time)
print(cp.asnumpy(s).sum())
print("CuPy version: ", cp.__version__)
