using Gen
import LinearAlgebra: diagm, I

include("helper_dists.jl")


###
### β priors
###

@gen function generalized_double_pareto_prior(J, σ_squared, g, ν::Real=1., ϑ::Real=1.)
    κs = Real[]
    for (k, j) in enumerate(J)
        if j
            ϕ = {(:ϕ, k)} ~ gamma(ν, ϑ)
            κ = {(:κ, k)} ~ exponential(ϕ^2/2)
            push!(κs, κ)
        end
    end
    β ~ mvnormal(zeros(length(κs)), diagm(σ_squared ./ (g .* κs)))
end

@gen function horseshoe_prior(J, σ_squared, g, ν::Union{Real,Nothing}=1., ϑ::Real=1)
    κs = Real[]
    for (k, j) in enumerate(J)
        if j
            if isnothing(ν)
                ν = {(:ν, k)} ~ half_cauchy(ϑ)  # paper calls this a hyperparameter, but also samples it
            else
                ν = {(:ν, k)} ~ dirac_delta(ν)
            end
            κ = {(:κ, k)} ~ half_cauchy(ν)
            push!(κs, κ)
        end
    end
    β ~ mvnormal(zeros(length(κs)), diagm(σ_squared ./ (g .* κs)))
end

@gen function normal_exponential_gamma_prior(J, σ_squared, g; ν::Real=1., ϑ::Real=1)
    κs = Real[]
    for (k, j) in enumerate(J)
        ζ_squared = {(:ζ_squared, k)} ~ gamma(ν, ϑ)
        κ = {(:κ, k)} ~ exponential(ζ_squared/2)
        push!(κs, κ)
    end
    β ~ mvnormal(zeros(length(κs)), diagm(σ_squared ./ (g .* κs)))
end


###
### Model
###

struct LMEModelHypers
    p0::Real
    pk0::Real
    μr0::AbstractVector{<:Real}
    Σr0::AbstractMatrix{<:Real}
end

"""
    lme_model(X, Z, hypers[, beta_prior])

Linear Mixed Effects model, with input `X[i,j]`,
and design matrix `Z[i,j,:]` such that (abstractly):

    y[i,j] = X[i,j,:]'β + Z[i,j,:]'ς[i,:] + ϵ[i,j]

where `β` is the vector of fixed effect coefficients,
`ς[i,:]` is the `i`th vector of random effects,
and `ϵ[i,j]` is the residual error.

More detail can be found in the paper:

    Yang, M., Wang, M. & Dong, G. Bayesian variable selection for mixed effects model with shrinkage prior. Comput Stat 35, 227–243 (2020). https://doi.org/10.1007/s00180-019-00895-x
"""
@gen function lme_model(X::AbstractArray{Real, 3},
                        Z::AbstractArray{Real, 3},
                        hypers::LMEModelHypers,
                        β_prior::GenerativeFunction=horseshoe_prior)
    l = size(X, 3)
    q = size(Z, 3)
    J = Bool[({(:J, k)} ~ bernoulli(hypers.p0)) for k in 1:l]
    n = Int[sum([isassigned(X[i,:,1], j) for j in 1:size(X, 2)]) for i in 1:size(X, 1)]

    # fixed effects coefficients
    σ_squared ~ inv_gamma(1, 1/2)
    g ~ gamma(1/2, sum(n)/2)
    βJ = {*} ~ β_prior(J, σ_squared, g)

    # random effects variables
    λ = Vector{Real}(undef, q)
    for k in 1:q
        ϕ_squared = {(:ϕ_squared, k)} ~ inv_gamma(1/2, 1/2)
        λ[k] = {(:λ, k)} ~ ZINplus([hypers.pk0, 1 - hypers.pk0], 0, ϕ_squared)
    end
    Λ = diagm(λ)
    Γ = Matrix{Real}(I(q))
    r_idx = 1
    for f in 2:q
        for m in 1:f-1
            if λ[f] == 0
                γ = {(:γ, f, m)} ~ dirac_delta(0)
            else
                # for simplicity, assume uncorrelated γ
                γ = {(:γ, f, m)} ~ normal(hypers.μr0[r_idx], hypers.Σr0[r_idx,r_idx])
            end
            Γ[f,m] = γ
            r_idx += 1
        end
    end

    # shrinkage prior over random effects
    ξ = Matrix{Real}(undef, size(X, 1), q)
    for i in 1:size(X, 1)
        for k in 1:q
            m = {(:m, i, k)} ~ gamma(1.0, 1.0)
            τ = {(:τ, i, k)} ~ exponential(m^2/2)
            ξ[i,k] = {(:ξ, i, k)} ~ normal(0, τ)
        end
    end

    # observation model
    y = Matrix{Real}(undef, size(X, 1), size(X, 2))
    for i in 1:size(X, 1)
        for j in 1:n[i]
            y[i,j] = {(:y, i, j)} ~ normal(X[i,j,J]'βJ + Z[i,j,:]'Λ*Γ*ξ[i,:], σ_squared)
        end
    end

    J, σ_squared, g, βJ, Λ, Γ, ξ, y
end


###
### Simulation
###

function simulate_data()
    n_subjects = 100
    n_obs = 10
    q = 8
    len_r = Int(q*(q-1)/2)
    hypers = LMEModelHypers(0.5, 0.5, zeros(len_r), I(len_r))

    # making X
    Xl = fill(1., n_subjects, n_obs, 1)
    Xr = Array{Real}(undef, n_subjects, n_obs, q-1)
    for n in eachindex(Xr)
        Xr[n] = uniform(-1, 1)
    end
    X = cat(Xl, Xr, dims=3)
    Z = X[:,:,1:4]
    σ = 0.5

    # making β
    β = Real[0.,0.,0.,0.,0.,0.,2.,2.]

    # making random effects
    Ω = [1e-6 0.00 0.00 0.00;
         0.00 0.90 0.48 0.06;
         0.00 0.48 0.40 0.10;
         0.00 0.06 0.10 0.10]
    ς = Matrix{Real}(undef, n_subjects, size(Z,3))
    for i in 1:n_subjects
        ς[i,:] = mvnormal(zeros(size(Z,3)), Ω)
    end

    # making observations and constraints
    constraints = choicemap()
    y = Matrix{Real}(undef, size(X, 1), size(X, 2))
    for i in 1:n_subjects
        for j in 1:n_obs
            y[i,j] = normal(X[i,j,:]'β + Z[i,j,:]'ς[i,:], σ^2)
            constraints[(:y, i, j)] = y[i,j]
        end
    end

    return X, Z, hypers, constraints
end
