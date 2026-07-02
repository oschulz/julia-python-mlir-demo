from common import save_mlir  # caps GPU memory before importing jax; provides save_mlir / load_mlir

import jax
import jax.numpy as jnp


x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
list(x.devices())

jax.jit(jnp.sum).lower(x).as_text()

# jaxpr: JAX's typed IR, one level above StableHLO
jax.make_jaxpr(jnp.sum)(x)


from userfuncs import negsumsq

jax.make_jaxpr(negsumsq)(x)           # jaxpr: integer_pow, reduce_sum, neg
jax.jit(negsumsq).lower(x).as_text()  # ... then lowered to StableHLO

save_mlir("negsumsq_jax.mlir", negsumsq, x)
