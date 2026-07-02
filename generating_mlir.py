from common import save_mlir  # caps GPU memory before importing jax; provides save_mlir / load_mlir

import jax
import jax.numpy as jnp


x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
list(x.devices())

jax.jit(jnp.sum).lower(x).as_text()


from userfuncs import negsumsq

jax.jit(negsumsq).lower(x).as_text()

save_mlir("negsumsq_jax.mlir", negsumsq, x)
