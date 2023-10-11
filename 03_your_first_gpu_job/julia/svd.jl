using CUDA
N = 8000
F = CUDA.svd(CUDA.rand(N, N))
println(sum(F.S))
println("completed")
