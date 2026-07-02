# Using Reactant (powered by Polygeist and Enzyme-JAX) to raise GPU kernels
# to MLIR.

using KernelAbstractions
import CUDA
include("common.jl")   # caps GPU memory before the first `using Reactant`

using Reactant
using Reactant: to_rarray, @code_hlo, @jit

@kernel function custom_contraction_kernel!(C, A, B)
    i, j = @index(Global, NTuple)
    acc = zero(eltype(C))
    for k in 1:size(A, 2)
        @inbounds acc += A[i, k] * B[k, j]
    end
    @inbounds C[i, j] = acc
end

function custom_contraction(A::AbstractMatrix, B::AbstractMatrix)
    I, K = size(A)
    K2, J = size(B)
    @assert K == K2 "contraction dim mismatch: A's last dim $K vs B's first dim $K2"
    C = similar(A, I, J)
    backend = KernelAbstractions.get_backend(A)
    kernel! = custom_contraction_kernel!(backend)
    kernel!(C, A, B; ndrange=size(C))
    KernelAbstractions.synchronize(backend)
    return C
end

I, K, J = 12, 30, 56
A = rand(Float32, I, K)
B = rand(Float32, K, J)
C_ref = A * B   # correctness reference

Ar = to_rarray(A)
Br = to_rarray(B)

# No code raising, calls CUDA kernel directly
@code_hlo raise = false custom_contraction(Ar, Br)
Cr = @jit raise = false custom_contraction(Ar, Br)

# Code raising to MLIR/StableHLO
@code_hlo raise = true custom_contraction(Ar, Br)
Cr_raised = @jit raise = true custom_contraction(Ar, Br)
