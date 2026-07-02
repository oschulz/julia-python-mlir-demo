ENV["XLA_REACTANT_GPU_MEM_FRACTION"] = "0.18"   # cap GPU memory before the first `using Reactant`

using Reactant: Reactant, XLA, @code_hlo, @jit, to_rarray
using InteractiveUtils: code_llvm

"""
    save_mlir(path, f, args...; optimize=true)

Compile `f(args...)` to StableHLO/MLIR with `@code_hlo` and write the text to
`path`. Counterpart of load_mlir in common.py.

`optimize` mirrors `@code_hlo`'s own keyword (see using_mlir.jl for the run-only-vs-optimized
comparison it's meant for). For modules headed to the PYTHON side: keep the default
`optimize=true` — it emits a native `stablehlo.fft`. `optimize=false` is WORSE there: it
unwinds FFT lowering into an `enzyme.batch` op, a whole extra dialect `enzyme_ad.jax.hlo_call`'s
parser doesn't know at all. The optimized form's own portability wrinkle (an
`enzymexla.complex_is_purely_real/imaginary = [#enzymexla<guaranteed ...>]` analysis hint on
FFT/slice ops — harmless for `Reactant.Ops.hlo_call`, which knows the dialect, but rejected by
`enzyme_ad.jax.hlo_call`'s parser as "unregistered dialect") is handled on the load side, in
common.py's `load_mlir_enzyme` — see its docstring.
"""
function save_mlir(path, f, args...; optimize::Bool=true)
    code = optimize ? string(@code_hlo f(args...)) : string(@code_hlo optimize=false f(args...))
    write(path, code)
    return nothing
end

"""
    save_llvm(path, f, args...; optimize=true)

Compile `f(args...)` to native-code LLVM IR with `code_llvm` and write the text to
`path` (conventionally a `.ll` file). Plain Julia/LLVM, not Reactant/MLIR — useful to
contrast against `save_mlir`'s StableHLO output for the same function.
"""
function save_llvm(path, f, args...; optimize::Bool=true)
    code = sprint(io -> code_llvm(io, f, Base.typesof(args...); optimize))
    write(path, code)
    return nothing
end

"""
    load_mlir(path)::Function

Read a StableHLO/MLIR module from `path` and return a Julia function that calls
its @main via `Reactant.Ops.hlo_call`.
"""
function load_mlir(path)
    code = read(path, String)
    return function (args...)
        mod_args = dim_reorder(args)
        result = only(Reactant.Ops.hlo_call(code, mod_args...))
        return dim_reorder(result)
    end
end

dim_reorder(x::Number) = x
dim_reorder(x::AbstractVector{<:Number}) = x
dim_reorder(x::AbstractMatrix{<:Number}) = permutedims(x, (2,1))
dim_reorder(x::AbstractArray{<:Number,N}) where {N} = permutedims(x, ntuple(i -> N - i + 1, N))
dim_reorder(x::Tuple) = map(dim_reorder, x)
dim_reorder(x::NamedTuple) = map(dim_reorder, x)
