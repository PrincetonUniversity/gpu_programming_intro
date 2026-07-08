using CUDA, LinearAlgebra
N = 8000
A = CUDA.rand(N, N)
F = svd(A)
println(sum(F.S))
println("completed")
