using CUDA
N = 8000
CUDA.svd(CUDA.rand(N, N))
