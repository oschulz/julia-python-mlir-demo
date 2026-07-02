import jax.numpy as jnp

def negsumsq(x):
    # A trivial scalar "loss": negative sum of squares. Gradient is -2x.
    return -jnp.sum(x**2)
