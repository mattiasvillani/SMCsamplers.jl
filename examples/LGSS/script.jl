# # Linear Gaussian state space model

# In this example we explore the joint posterior of the state $x_t$ in a simple linear gaussian state space (LGSS) model with known static parameters: 
#
# ```math
# \begin{align*}
#   y_t &= x_t + \epsilon_t, \quad \epsilon_t \sim N(0,\sigma_e) \\
#   x_t &= ax_{t-1} + \nu_t, \quad \nu_t \sim N(0,\sigma_v) \\
#   x_0 &\sim N(0, \sigma_v/\sqrt{1-a^2})  
# \end{align*}
# ```

# Two algorithms are compared: 
# 1. Particle Gibbs with Ancestor Sampling (PGAS)
# 2. Forward Filtering Backward Sampling (FFBS)

# First some preliminaries.
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random

colors = [
    "#6C8EBF", "#c0a34d", "#780000", "#007878",     
    "#b5c6df","#eadaaa","#AE6666", "#4CA0A0","#bf9d6c", "#3A6B35", 
    "#9d6a6d","#d9c6c7", "#98bbb9", "#bf8d6c", 
    "#CBD18F"]

gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

myquantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)
Random.seed!(123);

# ### Set up the state-space model
# Set up static parameter structure and the model distributions for PGAS, and set static parameter values
mutable struct LGSSParams 
    a::Float64
    σᵥ::Float64
    σₑ::Float64
end

prior(θ) = Normal(0, θ.σᵥ / √((1 - θ.a^2)));
transition(θ, state, t) = Normal(θ.a * state, θ.σᵥ);  
observation(θ, state, t) = Normal(state, θ.σₑ);   

a = 0.9         # Persistence
σᵥ = 0.3        # State std deviation
σₑ = 0.5        # Observation std deviation
θ = LGSSParams(a, σᵥ, σₑ); # Set up parameter struct for PGAS

# ### Simulate data from LGSS model
T = 200     # Length of time series
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

# ### PGAS sampling
Nₚ = 20      # Number of particles for PGAS
Nₛ = 1000;   # Number of samples from posterior
PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation)
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = myquantile(PGASdraws, [0.025, 0.975], dims = 3);

# ### FFBS Sampling
## Set up the LGSS for FFBS and sample
Σₑ = [θ.σₑ^2]
Σₙ = [θ.σᵥ^2]
μ₀ = [0;;]
Σ₀ = [θ.σᵥ^2/(1-θ.a^2);;]
A = θ.a
C = 1
B = 0
U = zeros(T,1);
FFBSdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, Nₛ);
FFBSmean = mean(FFBSdraws, dims = 3)[2:end,:,1] # Exclude initial state at t=0
FFBSquantiles = myquantile(FFBSdraws, [0.025, 0.975], dims = 3)[2:end,:,:];

# ### Plot the posterior mean and 95% C.I. intervals from both algorithms
plottrue = true
p = length(prior(θ))
plt = []
for j in 1:p

    #True state evolution
    if plottrue
        plt_tmp = plot(x, c = colors[3], lw = 1, label = "true state")
    else
        plt_tmp = plot()
    end

    #PGAS
    plot!(PGASmean[:,j], lw = 1,
        c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 

    #FFBS
    plot!(FFBSmean[:,j]; color = :black, lw = 1, linestyle = :dash, label="FFBS")
    plot!(FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label="")
    plot!(FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")

    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,p), size = (800,300))

