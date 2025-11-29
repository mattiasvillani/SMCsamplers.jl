# # Time-varying Poisson model

# In this example we analyze the time--varying Poisson Regression model with measurement model
# ```math
# \begin{align*}
#   y_t &\sim \mathrm{Poisson}(\exp(z_t^\top\boldsymbol{\beta}_t)) \\
#    \boldsymbol{\beta}_t &=  \boldsymbol{\beta}_{t-1} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim N(0,\boldsymbol{\Sigma}_\eta) \\
#   \boldsymbol{\beta}_0 &\sim N(0, \boldsymbol{\Sigma}_0)  
# \end{align*}
# ```

# ### Loading some packages and setting up the plotting style
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, ForwardDiff, PDMats
using LinearAlgebra, Measures

colors = ["#6C8EBF", "#c0a34d", "#780000", "#007878",  "#b5c6df", "#eadaaa"]

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

Random.seed!(123);

# ### Set up Poisson model
mutable struct ParamTvPoissonReg
    μ₀::Vector{Float64}
    Σ₀::PDMat{Float64}
    Σₙ::PDMat{Float64}
    Z::Vector{Matrix{Float64}} # Covariates for each group
end

prior(θ) = MvNormal(θ.μ₀, θ.Σ₀)
transition(θ, state, t) = MvNormal(state, θ.Σₙ)  
function observation(θ, state, t) # time t is group time
    λs = exp.(θ.Z[t]*state) # θ.Z[t] is a n_t × p matrix
    return product_distribution(Poisson.(λs)) # this is a multivariate dist
end

# Define a function to simulate data from the time-varying Poisson regression model and plot it
function simTvPoisReg(observation, transition, prior, μ₀, Σₙ, T, p; plotdata = true)

    β = zeros(T, p)
    β₀ = μ₀
    Z = randn(T, p)
    y = zeros(Int, T)
    for t in 1:T
        if t == 1
            β[t,:] = rand(MvNormal(β₀, Σₙ))
        else
            β[t,:] = rand(MvNormal(β[t-1,:], Σₙ))
        end
        y[t] = rand(Poisson(exp(Z[t,:] ⋅ β[t,:])))
    end 

    plt = nothing
    if plotdata

    end
    return β, y, Z, plt
end

# Define a function to plot the posterior distributions
function plotEvolDistributions!(plt, postDraws, quantiles, trueEvol = nothing, 
        label = nothing, shaded = false; plotSettings...)

    T, nState, nSim = size(postDraws)
    postquantiles = quantile_multidim(postDraws, quantiles, dims = 2);
    
    if !isnothing(trueEvol)
        plot!(plt, 1:T, trueEvol, lw = 1, c = :black, label = "true"; plotSettings...)
    end

    if shaded
        plot!(plt, 1:T, postquantiles[:,2], fillrange = postquantiles[:,1],
            fillalpha = 0.2, fillcolor = :gray, label = "", lw = 0) 
        plot!(plt, 1:T, postquantiles[:,2], fillrange = postquantiles[:,3],
            fillalpha = 0.2, fillcolor = :gray, label = "", lw = 0) 
        plot!(plt, 1:T, postquantiles[:,2], lw = 1, c = :gray, linestyle = :solid, 
            label = label)
    else
        plot!(plt, 1:T, postquantiles[:,1], label = ""; lw = 1, linestyle = :dash, 
            plotSettings...) 
        plot!(plt, 1:T, postquantiles[:,3], label = ""; lw = 1, linestyle = :dash,    
            plotSettings...) 
        plot!(plt, 1:T, postquantiles[:,2], label = label; lw = 1, 
            linestyle = :solid, plotSettings...)
    end

    return plt

end

# ### Simulate data time-varying Poisson model

# Setting up model parameters
p = 2      # State size - number of β parameters, including intercept
T = 200    # Number of observations
μ₀ = zeros(p)
Σ₀ = PDMat([10.0 0;0 10])
Σₙ = PDMat([0.01 0;0 0.01])     # State innovation covariance matrix

# Simulate some data and plot
β, y, Z, plt = simTvPoisReg(observation, transition, prior,  μ₀, Σₙ, T, p);
λ = exp.([Z[t,:] ⋅ β[t,:] for t = 1:T]) # True intensity
# Plot the true parameter evolution and the time series
plt1 = plot()
for j in 1:p
    plot!(plt1, β[:,j]; label= L"\beta_{%$(j-1)}", c = colors[j], xlab ="t")
end
plt2 = plot(1:T, log.(λ); label = L"\log\lambda_t", lw = 2, c = colors[2], 
    title ="log intensity", legend = :topright)
plt3 = plot(1:T, λ; label = L"\lambda_t", lw = 2, c = colors[2], 
    title ="intensity", ylims = (0, maximum(λ)*1.1), legend = :topright)
plt4 = plot(1:T, y; label = "data", lw = 2, c = colors[1], xlabel = "time", 
    ylabel = "", legend = false, title = "time series")
plt = plot(plt1, plt2, plt3, plt4, layout = (2,2), size = (800, 600), bottommargin = 5mm)

Y, Zg = splitEqualGroups(y, Z, 1)
θ = ParamTvPoissonReg(μ₀, Σ₀, Σₙ, Zg); # Instantiate the parameter struct for PGAS

# ### PGAS sampling
nSim = 1000;             #  Number of samples from posterior
nParticles = 100         # Number of particles for PGAS
sample_t0 = true         # Sample state at t=0 ?
PGASdraws = PGASsampler(Y, θ, nSim, nParticles, prior, transition, observation; 
    sample_t0 = sample_t0);

# Plot the true evolution and the posterior distributions from PGAS
quantiles = [0.025, 0.5, 0.975]

plts = []
for j = 1:p
    push!(plts, plot(; title = L" \beta_%$t"))
end
for j = 1:p
    plotEvolDistributions!(plt, postDraws, quantiles, trueEvol = nothing, 
            label = nothing, shaded = false; plotSettings...)
end

pltLogIntensity = plot(; title = "Log Intensity "*L" \log\lambda_t = x_t")
plotEvolDistributions!(pltLogIntensity, PGASdraws, quantiles, [NaN*ones(sample_t0);x], 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :bottomleft)

pltIntensity = plot(; title = "Intensity "*L" \lambda_t = \exp(x_t)")
plotEvolDistributions!(pltIntensity, exp.(PGASdraws), quantiles, 
    exp.([NaN*ones(sample_t0);x]), "PGAS(N=$nParticles)", true; color = colors[3], 
    lw = 1, legend = :topleft)

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
LaplaceDraws = FFBS_laplace(U, y, A, B, Σₙ, μ₀, Σ₀, observation, θ, nSim; 
    sample_t0 = sample_t0);

# Plot the PGAS and Laplace posterior distributions, this time without true evolution
pltLogIntensity = plot(; title = "Log Intensity "*L" \log\lambda_t = x_t", 
    legend = :bottom)
plotEvolDistributions!(pltLogIntensity, PGASdraws, quantiles, nothing, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1)

pltIntensity = plot(; title = "Intensity "*L" \lambda_t = \exp(x_t)")
plotEvolDistributions!(pltIntensity, exp.(PGASdraws), quantiles, nothing, 
    "PGAS(N=$nParticles)", true; color = colors[3], lw = 1, legend = :topleft)

plotEvolDistributions!(pltLogIntensity, LaplaceDraws[2:end,:,:], quantiles, nothing, 
    "Laplace", false; color = colors[3])
plotEvolDistributions!(pltIntensity, exp.(LaplaceDraws[2:end,:,:]), quantiles, nothing, 
    "Laplace", false; color = colors[3], lw = 1, legend = :topleft)

plot(pltLogIntensity, pltIntensity, layout = (1,2), size = (800, 300), bottommargin = 5mm)

# The posteror distribution from the Laplace approximation is very close to the posterior from PGAS.

## Grouped data 
function observation(θ, state, t)               # time t is now group time
    λs = exp.(θ.Z[t]*state)                     # θ.Z[t] This is a n_t × p matrix
    return product_distribution(Poisson.(λs))
end