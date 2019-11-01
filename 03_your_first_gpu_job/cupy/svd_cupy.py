import numpy as np
import cupy as cp

N = 1000
X = cp.random.randn(N, N, dtype=np.float64)
u, s, v = cp.linalg.decomposition.svd(X)
print(cp.asnumpy(s).sum())
print("CuPy version: ", cp.__version__)
