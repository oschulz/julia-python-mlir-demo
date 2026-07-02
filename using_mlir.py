# Load Reactant.jl-generated StableHLO/MLIR and use it in Python

from common import load_mlir, load_mlir_enzyme  # caps GPU memory before importing jax

import numpy as np
import jax
import jax.numpy as jnp


x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# direct jax, run only:
loss = load_mlir("negsumsq_reactant.mlir")
loss(x)

# enzyme-ad, can run and autodiff
loss_e = load_mlir_enzyme("negsumsq_reactant.mlir")
scalar = lambda v: loss_e(v)[0]
xj = jnp.asarray(x)
float(jax.jit(scalar)(xj))
jax.jit(jax.grad(scalar))(xj)
