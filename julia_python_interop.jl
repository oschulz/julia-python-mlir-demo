include("common.jl")

using Reactant
using Reactant: to_rarray

using PythonCall
pyimport("sys").path.insert(0, @__DIR__)

jax = pyimport("jax")
jnp = pyimport("jax.numpy")

x = Float32[1, 2, 3]
rx = to_rarray(x)

Reactant.XLA.device(rx)

@code_hlo jax.numpy.sum(rx)

userfuncs = pyimport("userfuncs")

@code_hlo userfuncs.negsumsq(rx)
