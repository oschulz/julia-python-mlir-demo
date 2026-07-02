# Load JAX-generated StableHLO/MLIR and use it in Julia

include("common.jl")            # caps GPU memory before the first `using Reactant`; defines load_mlir

using Reactant
using Reactant: to_rarray
using Enzyme

f_loss = load_mlir("negsumsq_jax.mlir")

x = Float32[1, 2, 3]
rx = to_rarray(x)

# Note the difference between the loaded MLIR code and the Reactant-optimized
# MLIR code:
@code_hlo optimize=false f_loss(rx)
@code_hlo f_loss(rx)

loss = @jit f_loss(rx)

# We can still differentiate loaded MLIR with Enzyme:
@code_hlo Enzyme.gradient(Enzyme.Reverse, f_loss, rx)
∇loss = @jit Enzyme.gradient(Enzyme.Reverse, f_loss, rx)
