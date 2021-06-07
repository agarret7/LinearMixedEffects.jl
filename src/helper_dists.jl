using Gen


struct DiracDelta <: Distribution{Real} end
const dirac_delta = DiracDelta()
Gen.logpdf(::DiracDelta, v::Real, x::Real) = v == x ? 0.0 : -Inf
Gen.random(::DiracDelta, x::Real) = x
Gen.has_output_grad(::DiracDelta) = false
Gen.has_argument_grads(::DiracDelta) = false

struct HalfNormal <: Distribution{Real} end
const half_normal = HalfNormal()
Gen.logpdf(::HalfNormal, x::Real, std::Real) = (@assert x >= 0; Gen.logpdf(normal, x, 0, std))
Gen.random(::HalfNormal, std::Real) = abs(normal(0, std))
Gen.has_argument_grads(::HalfNormal) = false

struct HalfCauchy <: Distribution{Real} end
const half_cauchy = HalfCauchy()
Gen.logpdf(::HalfCauchy, x::Real, gamma::Real) = (@assert x >= 0; Gen.logpdf(cauchy, x, 0, gamma))
Gen.random(::HalfCauchy, gamma::Real) = abs(cauchy(0, gamma))
Gen.has_argument_grads(::HalfNormal) = false

"""
Zero-inflated truncated positive normal prior
"""
ZINplus = HeterogeneousMixture([dirac_delta, half_normal])
