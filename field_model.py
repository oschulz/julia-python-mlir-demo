# Python counterpart of field_model.jl: load the two StableHLO/MLIR modules it saves:
#   field_model.mlir     expected_data: xi (harmonic latents, gp_dim) -> lambda (rate, data_dim)
#   field_model_fi.mlir   poisson_fi_diag: lambda -> Poisson Fisher info diag (same shape)
# and build the same inv_posterior_cov_operator

# KNOWN UPSTREAM LIMITATION: reverse-mode (VJP) through the loaded field_model.mlir fails —
# enzyme-ad's reverse-mode lowering of this sliced complex-FFT graph emits a malformed
# `complex.conj` op ("operand #0 must be complex type with floating-point elements, but got
# 'tensor<120x60xcomplex<f32>>'"). Verified independent of API entry point (same failure via
# jax.vjp and via jax.grad on a seeded scalar reduction) and independent of @code_hlo's
# optimize setting — a genuine enzyme-ad bug, not something fixable from here. Forward-mode
# (JVP) and the (non-FFT) Fisher-info module both work fine. See inv_posterior_cov_operator's
# try/except below for how this surfaces.

from common import load_mlir_enzyme

import jax
import jax.numpy as jnp

XI_SHAPE = (240, 120)      # Julia gp_dim   (120, 240), axis-reversed
LAMBDA_SHAPE = (120, 60)   # Julia data_dim (60, 120),  axis-reversed

_expected_data_raw = load_mlir_enzyme("field_model.mlir")       # xi -> (lambda,)
_poisson_fi_diag_raw = load_mlir_enzyme("field_model_fi.mlir")  # lambda -> (fi,)


def expected_data(xi):
    """xi (XI_SHAPE) -> expected Poisson rate lambda (LAMBDA_SHAPE). The GP forward model,
    loaded from field_model.mlir (Julia's f_expectation)."""
    (lam,) = _expected_data_raw(xi)
    return lam


def poisson_fi_diag(lam):
    """lambda -> diag(1/lambda), the Poisson Fisher information. Loaded from
    field_model_fi.mlir (Julia's poisson_fi_diag)."""
    (fi,) = _poisson_fi_diag_raw(lam)
    return fi


def J_times_v(f, x, v):
    """J*v (forward-mode JVP) — mirrors field_model.jl's
    Enzyme.autodiff(Forward, f, Duplicated(x, v))."""
    _, Jv = jax.jvp(f, (x,), (v,))
    return Jv


def Jt_times_v(f, x, w):
    """J^T*w (reverse-mode VJP) — mirrors field_model.jl's
    Enzyme.gradient(Reverse, dot(w, ·) ∘ f, x)."""
    _, vjp_fn = jax.vjp(f, x)
    (Jtw,) = vjp_fn(w)
    return Jtw


def inv_posterior_cov_operator(f_expected, xi_mean, xi, fi_diag):
    """Sigma^-1 . xi = (I + J^T fi_diag J) . xi — the MGVI Fisher metric operator, same
    math as field_model.jl's inv_posterior_cov_operator, built from the same two loaded
    MLIR modules. Needs reverse-mode through f_expected (see KNOWN UPSTREAM LIMITATION
    above) — currently fails for this FFT-containing module."""
    J_xi = J_times_v(f_expected, xi_mean, xi)
    IJ_xi = fi_diag * J_xi
    JtIJ_xi = Jt_times_v(f_expected, xi_mean, IJ_xi)
    return xi + JtIJ_xi


key = jax.random.PRNGKey(42)
key, k1, k2 = jax.random.split(key, 3)
xi_mean = jax.random.normal(k1, XI_SHAPE, dtype=jnp.float32)
xi = jax.random.normal(k2, XI_SHAPE, dtype=jnp.float32)

lam = jax.jit(expected_data)(xi_mean)
print(f"forward model: xi {xi_mean.shape} -> expected data {lam.shape}")

fi_diag = jax.jit(poisson_fi_diag)(lam)
print(f"Fisher info diag -> shape {fi_diag.shape}")

Jv = jax.jit(lambda x, v: J_times_v(expected_data, x, v))(xi_mean, xi)
print(f"J . v_xi (forward-mode JVP) -> shape {Jv.shape}")
