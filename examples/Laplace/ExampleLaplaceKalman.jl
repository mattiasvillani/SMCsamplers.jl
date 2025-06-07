# Required packages for this example are: Distributions, Plots, ForwardDiff, LinearAlgebra, PDMats. 

# Laplace Kalman filter example:

# The model is:
#y = Gamma(α, θₜ)   where I fix α = 2.0.
#log(θₜ) = log(θ[t-1]) + εₜ, where εₜ ~ N(0, σ²), I fix σ = 0.05.

# Problem set-up is that I specify a trajectory for the states(θ) as a sine curve, 
# then I simulate data from the Gamma and try to reconstruct the state trajectories

# start by loading packages and functions.
include(joinpath(@__DIR__, "functionsLaplaceKalman.jl"))


# Simulate state trajectory
θ = exp.(2*sin.(1:0.1:15)); 
T=length(θ)
plotPar = plot(θ, label="True θ", lw=2, c=:red)

α = 2.0
y = zeros(T);
[y[t] = rand(Gamma(α, θ[t])) for t ∈ 1:T];  # Simulate observations from Gamma(α, θ)
plotData = plot(y, label="Simulated y", lw=2)

# How non-Gaussian is the likelihood?
plot(plot(0.1:0.01:20, pdf.(Gamma.(α, 0.1:0.01:20), y[4]), label="y₄, θ≈6.87"),
plot(0.1:0.01:7, pdf.(Gamma.(α, 0.1:0.01:7), y[20]), label="y₃₀, θ≈2"), 
plot(0.1:0.01:7, pdf.(Gamma.(α, 0.1:0.01:7), y[30]), label="y₃₀, θ≈0.25"),
layout=(1,3), size=(900,300)) 



########################

# Time for inference.

# Log likelihood is:
logLik(θ, y) = logpdf(Gamma(α, exp.(θ)), y)  # Example log-likelihood function   


# Setting up model parameters (regular Kalman filter matrices):
A = 1.0;      # State transition
Σₙ = 0.05;    # state noise variance (σ²). 
B = 0.0;      # No exogenous variables.
μ₀, Σ₀ = 2., 1.1;  # Start values.
Y=deepcopy(y)
U=zeros(size(Y))


# Lets try a run:
ffbs_out = laplace_FFBS(U, Y, A, B, Σₙ, μ₀, Σ₀, logLik)
plot(ffbs_out)
plot!(log.(θ))


# Lets do multiple runs to get a feel for the variability.
nSim=1000
bs = zeros(T, nSim);
[bs[:,i] = laplace_FFBS(U, Y, A, B, Σₙ, μ₀, Σ₀, logLik)[2:end] for i in 1:nSim];

m_bs = mean(bs, dims=2);
std_bs = std(bs, dims=2);
plotEstStates = plot(m_bs, label="Mean of samples", lw=2, legend=:bottom)
plot!(m_bs + 2std_bs, fill = m_bs - 2std_bs,  label="2 sd error bands", c=:lightgrey, alpha=0.5)
plot!(log.(θ), label="True θ", lw=2, c=:orange)


plot(plotEstStates, plotData, plotPar, layout=@layout[a; [b c]] , size=(800,600))

