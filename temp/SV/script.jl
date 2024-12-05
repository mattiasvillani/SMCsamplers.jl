# Uses PGAS to simulate from the posterior of the state in stochastic volatility (SV) model:
#   x₀ ∼ N(0, σ₀)
#   xₜ = a⋅xₜ₋₁ + νₜ, νₜ ∼ N(0,σᵥ)
#   yₜ = exp(xₜ/2)εₜ, εₜ ∼ N(0,1)

using SMCsamplers, Plots, Distributions, LaTeXStrings, Random

# Plot settings
gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

myquantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)
Random.seed!(123)

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
plt1 = plot(exp.(x/2); label="", xlabel="t", lw = 2, legend = :topleft, 
    color = colors[3], title = "Standard deviation")
plt2 = plot(y; seriestype=:scatter, label="", xlabel="t", markersize = 2,
    color = colors[1], markerstrokecolor = :auto, title = "Realized observations")
plot(plt1, plt2, layout = (2,1))

# Run the algorithms
println("Generating $Nₛ PGAS draws based on $Nₚ particles")
@time PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation)
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = myquantile(PGASdraws, [0.025, 0.975], dims = 3)
    

plottrue = true
p = length(prior(θ))
for j in 1:p
    # True state evolution
    if plottrue
        plt = plot(x, c = :gray, lw = 1, label = "true state")
    else
        plt = plot()
    end
    # PGAS
    plot!(PGASmean[:,j], lw = 1,
        c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 

end
plt