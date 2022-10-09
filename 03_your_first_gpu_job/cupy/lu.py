from time import perf_counter
import numpy as np
import cupy as cp
import cupyx.scipy.linalg

N = 10000
X = cp.random.randn(N, N, dtype=np.float64)

trials = 5
times = []
for _ in range(trials):
  start_time = perf_counter()
  lu, piv = cupyx.scipy.linalg.lu_factor(X, check_finite=False)
  cp.cuda.Device(0).synchronize()
  times.append(perf_counter() - start_time)

print("Execution time: ", min(times))
print("CuPy version: ", cp.__version__)
