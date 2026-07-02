include("common.jl")   # caps GPU memory, then defines save_mlir / load_mlir

using Enzyme: Enzyme
import ForwardDiff

x = Float32[1, 2, 3]
rx = to_rarray(x)

XLA.device(rx)

@code_hlo sum(rx)


include("userfuncs.jl")

negsumsq(x)

@code_llvm negsumsq(x)
save_llvm("negsumsq.ll", negsumsq, x)

ForwardDiff.gradient(negsumsq, x)

@code_hlo optimize=false negsumsq(rx)
@code_hlo negsumsq(rx)

save_mlir("negsumsq_reactant.mlir", negsumsq, rx)

loss = @jit negsumsq(rx)

@code_hlo Enzyme.gradient(Enzyme.Reverse, negsumsq, rx)

∇loss = @jit Enzyme.gradient(Enzyme.Reverse, negsumsq, rx)
