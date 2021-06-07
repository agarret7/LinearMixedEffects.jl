# LinearMixedEffects.jl
Linear Mixed Effects model, implemented in Gen.jl (gen.dev).

For input `X[i,j]`, and design matrix `Z[i,j,:]` the model is (abstractly):
```julia
    y[i,j] = X[i,j,:]'β + Z[i,j,:]'ς[i,:] + ϵ[i,j]
```

where `β` is the vector of fixed effect coefficients,
`ς[i,:]` is the `i`th vector of random effects,
and `ϵ[i,j]` is the residual error.

More detail can be found in the paper:

Yang, M., Wang, M. & Dong, G. Bayesian variable selection for mixed effects model with shrinkage prior. Comput Stat 35, 227–243 (2020). https://doi.org/10.1007/s00180-019-00895-x


# Run

Download dependencies with:
```shell
    julia --project -e "import Pkg; Pkg.instantiate()"
```

The following will run a 1000 steps of the Gibbs sampler for β, given some
simulated data from the first experiment in the paper, and print the change
in joint log probabilities:
```shell
    julia --project src/LinearMixedEffects.jl
```
