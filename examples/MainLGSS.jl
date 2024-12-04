# Simulates from the posterior of the state in linear gaussian state space (LGSS) model:
#   x₀ ∼ N(0, σᵥ/√(1-a²))
#   xₜ = axₜ₋₁ + νₜ, νₜ ∼ N(0,σᵥ)
#   yₜ = xₜ + εₜ, εₜ ∼ N(0,σₑ)
# Two algorithms are compared: 
# 1. Particle Gibbs with Ancestor Sampling (PGAS)
# 2. Forward Filtering Backward Sampling (FFBS)

using SMCsamplers, Plots, Distributions, LaTeXStrings, Random

# Plot settings
gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

myquantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)
Random.seed!(123)

# Set up LGSS model structure for PGAS
mutable struct LGSSParams 
    a::Float64
    σᵥ::Float64
    σₑ::Float64
end
prior(θ) = Normal(0, θ.σᵥ / √((1 - θ.a^2)))
transition(θ, state, t) = Normal(θ.a * state, θ.σᵥ)  
observation(θ, state, t) = Normal(state, θ.σₑ)   

# Set up Linear Gaussian State Space Model
a = 0.9         # Persistence
σᵥ = 0.3        # State std deviation
σₑ = 0.5        # Observation std deviation
T = 200         # Length of time series

# Algorithm settings
Nₚ = 20      # Number of particles for PGAS
Nₛ = 1000     # Number of samples from posterior

θ = LGSSParams(a, σᵥ, σₑ) # Set up parameter struct for PGAS

# Simulate data from LGSS model
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
plot(x; label="state, x", xlabel="t", lw = 2, legend = :topleft, color = colors[3])
plot!(y; seriestype=:scatter, label="observed, y", xlabel="t", markersize = 2,
    color = colors[1], markerstrokecolor = :auto)

# Run the algorithms
println("Generating $Nₛ PGAS draws based on $Nₚ particles")
@time PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation)
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = myquantile(PGASdraws, [0.025, 0.975], dims = 3)
    
## FFBS
# Set up state-space model for FFBS
Σₑ = [θ.σₑ^2]
Σₙ = [θ.σᵥ^2]
μ₀ = [0;;]
Σ₀ = [θ.σᵥ^2/(1-θ.a^2);;]
A = θ.a
C = 1
B = 0
U = zeros(T,1)

println("Generating $Nₛ FFBS draws")
@time FFBSdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, Nₛ);
FFBSmean = mean(FFBSdraws, dims = 3)[2:end,:,1] # Exclude initial state at t=0
FFBSquantiles = myquantile(FFBSdraws, [0.025, 0.975], dims = 3)[2:end,:,:]

plottrue = false
p = length(prior(θ))
plt = []
for j in 1:p
    # True state evolution
    if plottrue
        plt_tmp = plot(x, c = :gray, lw = 1, label = "true state")
    else
        plt_tmp = plot()
    end
    # PGAS
    plot!(PGASmean[:,j], lw = 1,
        c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 
    # FFBS
    plot!(FFBSmean[:,j]; color = :black, lw = 1, linestyle = :dash, label="FFBS")
    plot!(FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label="")
    plot!(FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")

    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,p), size = (800,300))