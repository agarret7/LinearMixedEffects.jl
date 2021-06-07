using Gen
import Memoize: @memoize

include("helper_dists.jl")

@memoize function info_matrix(X::Array{Real,3})
    X = reshape(X, size(X, 1) * size(X, 2), size(X, 3))
    X'X
end

@gen function β_proposal(trace)
    X, Z, hypers = get_args(trace)
    J, σ_squared, g, βJ, Λ, Γ, ξ, y = get_retval(trace)
    l = size(X, 3)
    q = size(Z, 3)

    K = info_matrix(X)[J,J]
    ζJ = diagm([trace[(:κ,k)] for k in 1:l if J[k]])
    ΣβJ = inv(K/σ_squared + g/σ_squared*inv(ζJ))
    μβJ = zeros(sum(J))  # TODO add actual equation for conditional posterior mean

    β ~ mvnormal(μβJ, ΣβJ)
end

@gen function J_proposal(trace)
    error("not implemented")
end

@gen function g_proposal(trace)
    error("not implemented")
end

@gen function σ_squared_proposal(trace)
    error("not implemented")
end

@gen function r_proposal(trace)
    error("not implemented")
end

@gen function λ_proposal(trace)
    error("not implemented")
end

@gen function ξ_proposal(trace)
    error("not implemented")
end

@gen function ϕ_squared_proposal(trace)
    error("not implemented")
end

@gen function κ_proposal(trace)
    error("not implemented")
end

@gen function φ_proposal(trace)
    error("not implemented")
end

@gen function τ_proposal(trace)
    error("not implemented")
end

@gen function m_proposal(trace)
    error("not implemented")
end

@kern function lme_gibbs(trace, n_iters::Int)
    for i in n_iters
        trace ~ mh(trace, β_proposal, ())          # (1)
        trace ~ mh(trace, J_proposal, ())          # (2)
        trace ~ mh(trace, g_proposal, ())          # (3)
        trace ~ mh(trace, σ_squared_proposal, ())  # (4)
        trace ~ mh(trace, r_proposal, ())          # (5)
        trace ~ mh(trace, λ_proposal, ())          # (6)
        trace ~ mh(trace, ξ_proposal, ())          # (7)
        trace ~ mh(trace, ϕ_squared_proposal, ())  # (8)
        trace ~ mh(trace, κ_proposal, ())          # (9)
        trace ~ mh(trace, φ_proposal, ())          # (10)
        trace ~ mh(trace, τ_proposal, ())          # (11)
        trace ~ mh(trace, m_proposal, ())          # (12)
    end
    return trace
end
