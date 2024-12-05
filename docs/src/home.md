```@meta
CurrentModule = SMCsamplers
```

# SMCsamplers.jl

## Description

This is the start of a package for some Sequential Monte Carlo samplers in Julia. Some examples scripts can be found in the `examples` folder.

## Installation
Install from the Julia package manager (via Github) by typing `]` in the Julia REPL:
```
] add git@github.com:mattiasvillani/SMCsamplers.jl.git
```

## Example
```julia
# PGAS to simulate from the posterior of the state in stochastic volatility (SV) model:
#   x₀ ∼ N(0,σ₀)
#   xₜ = a⋅xₜ₋₁ + νₜ, νₜ ∼ N(0,σᵥ)
#   yₜ = exp(xₜ/2)εₜ, εₜ ∼ N(0,1)

using SMCsamplers, Plots, Distributions, LaTeXStrings, Random

# Set up SV model structure for PGAS
mutable struct SVParams 
    a::Float64
    σᵥ::Float64
    σ₀::Float64
end
prior(θ) = Normal(0, θ.σ₀)
transition(θ, state, t) = Normal(θ.a * state, θ.σᵥ)  
observation(θ, state, t) = Normal(0, exp(state/2))

# Set model parameters
a = 0.9         # Persistence
σᵥ = 1          # State std deviation
σ₀ = 0.5        # Initial observation std deviation
T = 200         # Length of time series

θ = SVParams(a, σᵥ, σₑ) # Set up parameter struct for PGAS

# Algorithm settings
Nₚ = 20         # Number of particles for PGAS
Nₛ = 1000       # Number of samples from posterior

# Simulate data from SV model
x = zeros(T)
y = zeros(T)
x0 = rand(prior(θ))
for t in 1:T
    if t == 1
        x[t] = rand(transition(θ, x0, t))
    else
        x[t] = rand(transition(θ, x[t-1], t))
    end
    y[t] = rand(observation(θ, x[t], t))
end 

# Simulate from joint smoothing posterior using PGAS for given static parameters
PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation) # returns (T, 1, Nₛ) array

```

