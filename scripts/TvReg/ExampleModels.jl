
""" 
    y, Z, β, Q = SimTVReg(T, p, σₑ, Σₙ, μ = nothing, Φ = nothing, μ₀ = zeros(p), Σ₀ = Σₙ)

Simulates `T` observations from the time-varying regression model with `p` covariates.
If isnothing(μ) then the parameters innovations are homoscedastic with covariance matrix Σₙ, otherwise the parameter innovations follow a stoch volatility process:

yₜ = z_t'βₜ + εₜ, εₜ ∼ N(0,σₑ²)

βₜ = βₜ₋₁ + νₜ, νₜ ∼ MvNormal(0, Qₜ = exp(hₜ))

hₜ = hₜ₋₁ + ηₜ, ηₜ ∼ MvNormal(Σₙ)

with β₀ ~ N(μ₀, Σ₀)

""" 
function SimTVReg(T, p, σₑ, Σₙ, μ = nothing, Φ = nothing, μ₀ = zeros(p), Σ₀ = Σₙ)

    Z = randn(T, p) # Matrix with covariates
    hetero = isnothing(μ) ? false : true
    # Simulate h-process and set up covariance matrices for the state innovations
    Q = [Σₙ for t = 1:T]
    if hetero
        H = repeat(μ', T, 1)
        Q[1] = PDMat(diagm(exp.(μ)))  
        for t = 2:T
            H[t,:] = μ + Φ*(H[t-1,:] - μ) + rand(MvNormal(Σₙ))
            Q[t] = PDMat(diagm(exp.(H[t,:]))) 
        end
    end
    y = zeros(T)
    β = zeros(T, p)
    for t in 1:T
        if t == 1
            β₀ = rand(MvNormal(μ₀, Σ₀))
            β[t,:] = β₀ + rand(MvNormal(Q[t]))
        else
            β[t,:] = β[t-1,:] + rand(MvNormal(Q[t]))
        end
        y[t] = Z[t,:] ⋅ β[t,:] + rand(Normal(0, σₑ))
    end
    return y, Z, β, Q
end