# Load the TensorFlow SavedModels that saving_tensorflow.jl exported from Julia and
#   (1) run inference with the forward model, and
#   (2) TRAIN the model — the gradient was exported too (Enzyme-on-MLIR), so TensorFlow
#       can keep optimizing the weights with no Julia in the loop.
#
# The SavedModel is an opaque XLA program (tf.raw_ops.XlaCallModule wrapping Reactant's
# StableHLO); TF just executes it. Run saving_tensorflow.jl first to produce the models.
#
# Dimension convention: Reactant declares the compiled @main boundary row-major, so every
# array's shape is REVERSED versus Julia's column-major (a Julia (2,4) weight is (4,2) in
# TF). We transpose on the way in/out. This is pure TF — no jax/enzyme import here.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""       # keep TF on CPU (GPU belongs to the Julia session)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

base = os.path.dirname(os.path.abspath(__file__))

# --- 1. Inference with the forward model (weights baked in on the Julia side) ----------
fwd = tf.saved_model.load(os.path.join(base, "affine_savedmodel"))

W0 = np.loadtxt(os.path.join(base, "affine_init_W.csv"), dtype=np.float32).reshape(2, 4)
b0 = np.loadtxt(os.path.join(base, "affine_init_b.csv"), dtype=np.float32).reshape(2)
xj = np.loadtxt(os.path.join(base, "affine_x.csv"), dtype=np.float32).reshape(4, 8)
yj = np.loadtxt(os.path.join(base, "affine_y.csv"), dtype=np.float32).reshape(2, 8)

x_tf = tf.constant(xj.T)          # (8, 4): Julia (4, 8) transposed
# XlaCallModule adds a leading batch axis; squeeze it back to (batch, out) = (8, 2).
pred_tf = np.squeeze(np.asarray(fwd.f(x_tf)), axis=0)
pred_ref = xj.T @ W0.T + b0       # same affine map computed directly with the Julia weights
print("inference: TF forward matches direct compute?",
      np.allclose(pred_tf, pred_ref, rtol=1e-4), "  shape", pred_tf.shape)

# --- 2. Train the model using the exported gradient ------------------------------------
grad = tf.saved_model.load(os.path.join(base, "affine_grad_savedmodel"))

y_tf = tf.constant(yj.T)          # (8, 2)
W = tf.Variable(W0.T)             # (4, 2): live weights, TF's to update
b = tf.Variable(b0)               # (2,)

loss0, _, _ = grad.f(x_tf, y_tf, W, b)
print(f"initial loss (TF): {float(loss0):.6f}   (should match Julia's printed value)")

lr = 0.2
for step in range(200):
    loss, gW, gb = grad.f(x_tf, y_tf, W, b)   # gradients come from the exported StableHLO
    W.assign_sub(lr * gW)                     # plain gradient descent in TF
    b.assign_sub(lr * gb)
    if step % 40 == 0 or step == 199:
        print(f"  step {step:3d}  loss = {float(loss):.6f}")

print("\nTensorFlow trained a Julia-defined model through its Enzyme-exported gradient.")
