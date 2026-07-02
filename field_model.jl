# Core MGVI metric operations on a NIFTy-like Gaussian-process-based field
# model. Only demonstrates the central operation Σ⁻¹ * ξ = (Jᵀ ℐ_λ J + I) * ξ,
# implemented via Enzyme forward- and reverse-mode autodiff.

include("common.jl")

using Reactant
using Reactant: to_rarray, @code_hlo, @jit
using Enzyme
using FFTW
using LinearAlgebra
using Random

using Plots

# Discrete Hartley Transform (DHT), real→real, self-inverse
function dht(x)
    X = FFTW.fft(Complex.(x))
    return real.(X) .- imag.(X)
end

# Field model based on Gaussian process defined by power law in frequency space
struct FieldModel{IT<:Integer,FT<:Real} <: Function
    data_dim::Tuple{IT,IT}
    gp_dim::Tuple{IT,IT}
    corrlen::FT
end

FieldModel(; data_dim::Tuple{Integer,Integer}, pad_factor::Integer, corrlen::Real) =
    FieldModel(data_dim, pad_factor .* data_dim, Float32(corrlen))


n_xi(m::FieldModel) = prod(m.gp_dim)        # length of ξ (latent parameter space)
n_lambda(m::FieldModel) = prod(m.data_dim)  # length of λ (data space)

# data region within padded GP grid
function data_idx(m::FieldModel)
    off = (m.gp_dim .- m.data_dim) .÷ 2
    (off[1]+1:off[1]+m.data_dim[1], off[2]+1:off[2]+m.data_dim[2])
end

# Square root of the (mirror-symmetric) power spectrum
function sqrt_power_spectrum(m::FieldModel)
    kfreq(n) = Float32[(i <= n ÷ 2 ? i : i - n) for i in 0:n-1]   # 0,1,..,n/2,..,-2,-1 (like map_idx)
    kx, ky = kfreq(m.gp_dim[1]), kfreq(m.gp_dim[2])
    return Float32[exp(-0.5f0 * m.corrlen^2 * (kx[i]^2 + ky[j]^2)) for i in 1:m.gp_dim[1], j in 1:m.gp_dim[2]]
end


# forward model expectation (expected Poisson rate per pixel)
function f_expectation(m::FieldModel, ξ::AbstractMatrix)
    s = dht(sqrt_power_spectrum(m) .* ξ)     # correlated real field s (the actual GP)
    expected_data = exp.(s)[data_idx(m)...]  # positive expected rates (log-Gaussian, not the GP itself)
    return expected_data
end

# Make FieldModel objects callable:
(m::FieldModel)(ξ) = f_expectation(m, ξ)


# Standard multivariate normal prior on ξ, identity covariance and zero mean:
sample_prior(m::FieldModel, rng = Random.default_rng()) = randn(rng, Float32, m.gp_dim)   # MV std-normal ξ

# Instantiate a field model:
m = FieldModel(data_dim = (60, 120), pad_factor = 2, corrlen = 0.8f0)

plot(sqrt_power_spectrum(m)[begin,:])

ξ_truth = sample_prior(m)  # Dummy ground truth drawn from latent prior

λ_truth = f_expectation(m, ξ_truth)  # Expected data (Poisson rate) at ξ_truth
heatmap(λ_truth)

poisson_fi_diag(rates::AbstractArray) = 1f0 ./ rates  # Diagonal of Poisson Fisher information

# Enzyme-powered JVP and VJP operations:
J_times_v(f, x, v)  = only(Enzyme.autodiff(Enzyme.Forward, f, Duplicated(x, v)))
Jᵀ_times_v(f, x, v) = Enzyme.gradient(Enzyme.Reverse, Base.Fix1(dot, v) ∘ f, x)[1]

# Approximate inverse posterior covariance times ξ, given ℐ_diag
function inv_posterior_cov_operator(f_expected, ξ_mean, ξ, ℐ_diag)
    Jξ    = J_times_v(f_expected, ξ_mean, ξ)
    ℐJξ  = ℐ_diag .* Jξ
    JᵀℐJξ = Jᵀ_times_v(f_expected, ξ_mean, ℐJξ)
    return ξ + JᵀℐJξ  # Σ⁻¹ * ξ = (Jᵀ ℐ_λ J + I) * ξ
end


# Dummy mean and current ξ, drawn from prior and moved to Reactant device:
Random.seed!(42)
ξ_mean = to_rarray(sample_prior(m))
ξ = to_rarray(sample_prior(m))

# Test-run forward model and save MLIR/StableHLO:
λ = @jit m(ξ_mean)  # expected data, 
println("forward model: ξ ", size(ξ_mean), " → expected data ", size(λ))

save_mlir("field_model.mlir", m, ξ_mean)

ℐ = @jit poisson_fi_diag(λ)                      # Poisson Fisher info at ξ (λ-space, constant)

save_mlir("field_model_fi.mlir", poisson_fi_diag, λ)

# Test JVP and VJP:
Jv  = @jit J_times_v(m, ξ_mean, ξ)                 # shape data_dim
Jᵀv = @jit Jᵀ_times_v(m, ξ_mean, λ)                # shape gp_dim

@code_hlo J_times_v(m, ξ_mean, ξ)
@code_hlo Jᵀ_times_v(m, ξ_mean, λ)

# Apply inverse posterior covariance operator to ξ, given ℐ_diag:
Mv = @jit inv_posterior_cov_operator(m, ξ_mean, ξ, ℐ)
println("Σ⁻¹·v_ξ = (I + Jᵀ ℐ_λ J)·v_ξ   -> size ", size(Mv), " (ξ-space)")
println("ξ' Σ⁻¹ ξ = ", dot(Array(ξ), Array(Mv)), "  (> 0: metric is SPD)")

@code_hlo inv_posterior_cov_operator(m, ξ_mean, ξ, ℐ)
save_mlir("inv_pstr_cov_op.mlir", inv_posterior_cov_operator, m, ξ_mean, ξ, ℐ)


# Load forward model and fisher information function from MLIR and run:
m2 = load_mlir("field_model.mlir")
λ2 = @jit m2(ξ_mean)

poisson_fi_diag2 = load_mlir("field_model_fi.mlir")
ℐ2 = @jit poisson_fi_diag(λ2)

@jit inv_posterior_cov_operator(m2, ξ_mean, ξ, ℐ2)
@code_hlo inv_posterior_cov_operator(m2, ξ_mean, ξ, ℐ2)
save_mlir("inv_pstr_cov_op2.mlir", inv_posterior_cov_operator, m2, ξ_mean, ξ, ℐ2)
