module LinearMixedEffects

using Gen

include("model.jl")
include("inference.jl")

X, Z, hypers, constraints = simulate_data()
trace, _ = generate(lme_model, (X, Z, hypers), constraints)
display(get_choices(trace))

# "score" is just the joint probability
# joint probability p(θ,Y) where Y is the simulated data, and θ is sampled from the prior
init_score = get_score(trace)

println("Before MCMC:")
@show trace[:β]
println()

@timev for i in 1:1000
    global trace
    trace, _ = mh(trace, β_proposal, ())
end
println()

# final score should be greater, since β is now sampled from
# the conditional posterior p(β|θ\{β},Y), as opposed to before
# when it was just sampled from the prior p(β)
final_score = get_score(trace)

println("After MCMC:")
@show trace[:β]
@show final_score - init_score
println()

end # module
