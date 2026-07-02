# Export a Reactant-compiled function as a TensorFlow SavedModel — including its
# Enzyme-computed gradient, so TensorFlow can load the model and keep *training* it.
#
# Reactant compiles a Julia function to a StableHLO module, then serializes it into a
# TF SavedModel (via `Reactant.Serialization`, which wraps the StableHLO->TF path with
# `tf.raw_ops.XlaCallModule`). TensorFlow runs it as an opaque XLA program — it never
# sees Julia. Loading + running the result lives in `using_tensorflow.py`.
#
# TF SavedModel export needs PythonCall loaded (activates ReactantPythonCallExt) and an
# importable `tensorflow` in the Python env (see CondaPkg.toml). The export itself is
# device-independent (a portable StableHLO artifact); TF then runs it on CPU.

include("common.jl")            # caps GPU memory before the first `using Reactant`

using Reactant
using Reactant: to_rarray
using Enzyme
using Random
using DelimitedFiles

using PythonCall                # activates ReactantPythonCallExt -> enables SavedModel export
import Reactant.Serialization as RS

@assert RS.serialization_supported(Val(:SavedModel)) "TensorFlow SavedModel export not \
    available — is `tensorflow` installed in the CondaPkg env and PythonCall loaded?"

# A tiny trainable model: a linear regressor  ŷ = W*x + b  with mean-squared-error loss.
predict(x, W, b) = W * x .+ b
mse(x, y, W, b)  = sum(abs2, predict(x, W, b) .- y) / Float32(size(x, 2))

# Loss and gradients w.r.t. the parameters (W, b), via Enzyme reverse mode. x and y are
# Const (data, not differentiated); W and b are the active arguments.
function loss_and_grads(x, y, W, b)
    res = Enzyme.gradient(Enzyme.ReverseWithPrimal, mse, Const(x), Const(y), W, b)
    return res.val, res.derivs[3], res.derivs[4]   # (loss, ∂loss/∂W, ∂loss/∂b)
end

# Concrete data and initial parameters (features × batch layout):
Random.seed!(42)
n_feat, n_out, n_batch = 4, 2, 8
x = to_rarray(rand(Float32, n_feat, n_batch))
y = to_rarray(rand(Float32, n_out, n_batch))
W = to_rarray(rand(Float32, n_out, n_feat))
b = to_rarray(rand(Float32, n_out))

# --- 1. Export the forward model for INFERENCE (weights frozen as constants) ----------
# W and b are baked into the SavedModel as `Parameter`s (from state_dict); only x is a
# runtime input. This is the "deploy a trained model" case.
compiled_predict = @compile serializable=true predict(x, W, b)
ŷ = compiled_predict(x, W, b)
println("forward model: x ", size(x), " → ŷ ", size(ŷ))

RS.export_as_tf_saved_model(
    compiled_predict, joinpath(@__DIR__, "affine_savedmodel"), v"1.8.5",
    [ RS.TFSavedModel.InputArgument(1),   # x: runtime input
      RS.TFSavedModel.Parameter("W"),     # W: baked-in constant
      RS.TFSavedModel.Parameter("b") ],   # b: baked-in constant
    Dict("W" => W, "b" => b),
)

# --- 2. Export the gradient for TRAINING (weights are live runtime inputs) -------------
# All of (x, y, W, b) are runtime inputs (empty input_locations ⇒ all InputArgument), so
# TensorFlow can hold W, b as tf.Variables, get gradients from this function, and update
# them. Output is (loss, ∂W, ∂b).
compiled_grad = @compile serializable=true loss_and_grads(x, y, W, b)
loss0, gW, gb = compiled_grad(x, y, W, b)
println("initial loss = ", Float32(loss0), "   ∂W ", size(gW), "  ∂b ", size(gb))

RS.export_as_tf_saved_model(
    compiled_grad, joinpath(@__DIR__, "affine_grad_savedmodel"), v"1.8.5",
)

# Save the initial parameters and the (x, y) batch so using_tensorflow.py can train from
# the identical starting point and cross-check the initial loss across frameworks.
writedlm(joinpath(@__DIR__, "affine_init_W.csv"), Array(W))
writedlm(joinpath(@__DIR__, "affine_init_b.csv"), Array(b))
writedlm(joinpath(@__DIR__, "affine_x.csv"), Array(x))
writedlm(joinpath(@__DIR__, "affine_y.csv"), Array(y))

println("\nWrote affine_savedmodel/ (inference), affine_grad_savedmodel/ (training), and")
println("initial-state CSVs. Now run using_tensorflow.py to load and train them in TF.")
