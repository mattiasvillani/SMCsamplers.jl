# # Time-varying Poisson model

# In this example we analyze the time--varying Poisson model with measurement model
#
# ```math
# \begin{equation*}
#   y_t \sim \mathrm{Poisson}(\exp(z_t)),
# \end{equation*}
# ```
# and a linear Gaussian transition model:
# ```math
# \begin{equation*}
#   x_t = \mu + a (x_{t-1} - \mu) + \eta_t, \quad \eta_t \sim N(0, \sigma_\eta^2).
# \end{equation*}
# ```

# ### Loading some packages and setting up the plotting style
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, ForwardDiff, PDMats
using LinearAlgebra, Measures

colors = ["#6C8EBF", "#c0a34d", "#780000", "#007878",  "#b5c6df", "#eadaaa"]

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

myquantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)
Random.seed!(123);

# ### Set up Poisson model
mutable struct PoissonParams 
    a::Float64
    μ::Float64
    σᵥ::Float64
    σ₀::Float64
end
prior(θ) = Normal(0, θ.σ₀)
transition(θ, state, t) = Normal(θ.μ +  θ.a * (state - θ.μ), θ.σᵥ)  
observation(θ, state, t) = Poisson(exp(state[1]));  

# ### Define helper functions

# Define a function to simulate data from the time-varying Poisson model and plot it
function simTvPoisson(observation, transition, prior, θ, T; plotdata = true)

    x = zeros(T)
    y = zeros(Int, T)
    x0 = θ.μ
    for t in 1:T
        if t == 1
            x[t] = rand(transition(θ, x0, t))
        else
            x[t] = rand(transition(θ, x[t-1], t))
        end
        y[t] = rand(observation(θ, x[t], t))
    end 

    plt = nothing
    if plotdata

        plt1 = plot(exp.(x), label="true", xlabel="t", lw = 2, legend = :bottomleft, 
            color = colors[3], title = "intensity "*L"\lambda_t = \exp(x_t)")
        plot!(y, seriestype=:scatter, xlabel="t", markersize = 2, label = "data",
            color = colors[1], markerstrokecolor = :auto)
        offset = 1e-1  # Offset to avoid log(0) in the plots
        plt2 = plot(x, xlabel="t", lw = 2, legend = :bottomleft, label = "true", 
        color = colors[3], title = "log intensity "*L"\log \lambda_t = x_t")
        plot!(log.(y .+ offset); seriestype=:scatter, xlabel="t", markersize = 2,
            color = colors[1], markerstrokecolor = :auto, label = "log(data + offset)")

        plt = plot(plt1, plt2, layout = (1,2), size = (800, 300), bottommargin = 5mm)

    end
    return x, y, plt
end

# Define a function to plot the posterior distributions
function plotEvolDistributions!(plt, postDraws, quantiles, trueEvol = nothing, 
        label = nothing, shaded = false; plotSettings...)

    T, nState, nSim = size(postDraws)
    postquantiles = myquantile(postDraws, quantiles, dims = 3);
    
    if !isnothing(trueEvol)
        plot!(plt, 1:T, trueEvol, lw = 1, c = :black, label = "true"; plotSettings...)
    end

    if shaded
        plot!(plt, 1:T, postquantiles[:,1,2], fillrange = postquantiles[:,1,1],
            fillalpha = 0.2, fillcolor = :gray, label = "", lw = 0) 
        plot!(plt, 1:T, postquantiles[:,1,2], fillrange = postquantiles[:,1,3],
            fillalpha = 0.2, fillcolor = :gray, label = "", lw = 0) 
        plot!(plt, 1:T, postquantiles[:,1,2], lw = 1, c = :gray, linestyle = :solid, 
            label = label)
    else
        plot!(plt, 1:T, postquantiles[:,1,1], label = ""; lw = 1, linestyle = :dash, 
            plotSettings...) 
        plot!(plt, 1:T, postquantiles[:,1,3], label = ""; lw = 1, linestyle = :dash,    
            plotSettings...) 
        plot!(plt, 1:T, postquantiles[:,1,2], label = label; lw = 1, 
            linestyle = :solid, plotSettings...)
    end

    return plt

end

# ### Simulate data time-varying Poisson model

# Setting up model parameters
a = 0.8             # Persistence
μ = 1               # Unconditional log intensity  
σᵥ = 0.3            # State std deviation
σ₀ = 10             # Initial observation std deviation
θ = PoissonParams(a, μ, σᵥ, σ₀); # Instantiate the parameter struct for PGAS
T = 200;            # Length of time series

# Simulate some data and plot
x, y, plt = simTvPoisson(observation, transition, prior, θ, T);
plt

# ### PGAS sampling
nSim = 1000;             #  Number of samples from posterior
nParticles = 100         # Number of particles for PGAS
PGASdraws = PGASsampler(y, θ, nSim, nParticles, prior, transition, observation);

# Plot the true evolution and the posterior distributions from PGAS
quantiles = [0.025, 0.5, 0.975]
pltLogIntensity = plot(; title = "Log Intensity "*L" \log\lambda_t = x_t")
plotEvolDistributions!(pltLogIntensity, PGASdraws, quantiles, x, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topright)

pltIntensity = plot(; title = "Intensity "*L" \lambda_t = \exp(x_t)")
plotEvolDistributions!(pltIntensity, exp.(PGASdraws), quantiles, exp.(x), 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topright)

plot(pltLogIntensity, pltIntensity, layout = (1,2), size = (800, 300), bottommargin = 5mm)

# ### Laplace approximation

# Setting up model parameters
Σₙ = [θ.σᵥ^2]
μ₀ = [0]
Σ₀ = [θ.σ₀^2;;]
A = θ.a
B = θ.μ*(1-θ.a) # The transition model is x_t = μ + a (x_{t-1} - μ) + η_t, so B = μ*(1-a)
U = ones(T,1);

# Simulate from the Laplace approximation of the posterior distribution 
LaplaceDraws = FFBS_laplace(U, y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim);

# Plot the PGAS and Laplace posterior distributions, this time without true evolution
pltLogIntensity = plot(; title = "Log Intensity "*L" \log\lambda_t = x_t")
plotEvolDistributions!(pltLogIntensity, PGASdraws, quantiles, nothing, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topright)

pltIntensity = plot(; title = "Intensity "*L" \lambda_t = \exp(x_t)")
plotEvolDistributions!(pltIntensity, exp.(PGASdraws), quantiles, nothing, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topright)

plotEvolDistributions!(pltLogIntensity, LaplaceDraws[2:end,:,:], quantiles, nothing, 
    "Laplace", false; color = colors[3])
plotEvolDistributions!(pltIntensity, exp.(LaplaceDraws[2:end,:,:]), quantiles, nothing, 
    "Laplace", false; color = colors[3], lw = 1, legend = :topright)

plot(pltLogIntensity, pltIntensity, layout = (1,2), size = (800, 300), bottommargin = 5mm)

# The posteror distribution from the Laplace approximation is very close to the posterior from PGAS.

# ### Case with more zero counts in the data

# We will now try a more challenging case with λ = exp(μ) = exp(-1) ≈ 0.3678, giving many zero counts in the data.
μ = -1               # Unconditional log intensity  
θ = PoissonParams(a, μ, σᵥ, σ₀); # Set up parameter struct for PGAS
B = θ.μ*(1-θ.a)     # Modify the transition model for Laplace accordingly

# Simulate some data and plot
x, y, plt = simTvPoisson(observation, transition, prior, θ, T);
plt

# Simulate from the PGAS posterior distribution
PGASdraws = PGASsampler(y, θ, nSim, nParticles, prior, transition, observation);

# Simulate from the Laplace approximation of the posterior distribution 
LaplaceDraws = FFBS_laplace(U, y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim);

# Plot the PGAS and Laplace posterior distributions, without true evolution
pltLogIntensity = plot(; title = "Log Intensity "*L" \log\lambda_t = x_t")
plotEvolDistributions!(pltLogIntensity, PGASdraws, quantiles, nothing, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topright)

pltIntensity = plot(; title = "Intensity "*L" \lambda_t = \exp(x_t)")
plotEvolDistributions!(pltIntensity, exp.(PGASdraws), quantiles, nothing, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topright)

plotEvolDistributions!(pltLogIntensity, LaplaceDraws[2:end,:,:], quantiles, nothing, 
    "Laplace", false; color = colors[3])
plotEvolDistributions!(pltIntensity, exp.(LaplaceDraws[2:end,:,:]), quantiles, nothing, 
    "Laplace", false; color = colors[3], lw = 1, legend = :topright)

plot(pltLogIntensity, pltIntensity, layout = (1,2), size = (800, 300), bottommargin = 5mm)

# The Laplace approximation is not accurate early on, but it improves as the data accumulates.