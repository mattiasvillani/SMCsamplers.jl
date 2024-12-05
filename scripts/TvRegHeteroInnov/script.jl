# PGAS for time-varying regression model with heteroscedastic parameter innovations
# yₜ = z_t'βₜ + εₜ, εₜ ∼ N(0,σₑ²)
# βₜ = βₜ₋₁ + νₜ, νₜ ∼ N(0,Qₜ), with Qₜ a deterministic sequence of covariance matrices
# β₀ ~ N(0, Σ₀)

# Note: this example shows the importance of using a good proposal for the first state, rather than the rather vague prior.

using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, PDMats
using LinearAlgebra, Measures

include("ExampleModels.jl") # exports SimTVReg to simulate from the model

# Plot settings
gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize=8,
    xtickfontsize=8, ytickfontsize=8, xguidefontsize=8, yguidefontsize=8,
    titlefontsize = 10, markerstrokecolor = :auto)

myquantile(A, p; dims, kwargs...) = mapslices(x -> quantile(x, p; kwargs...), A; dims)
Random.seed!(1234)

# Simulate some data
p = 2      # State size
T = 500    # Number of observations
σₑ = 1
μ = -9*ones(p)
Σₙ = PDMat(diagm(exp.(μ))) 
Φ = 0.8*I(p)
μ₀ = zeros(p)
Σ₀ = PDMat([50 0;0 50]) 
y, Z, β, Q = SimTVReg(T, p, σₑ, Σₙ, μ, Φ, μ₀, Σ₀)

# Plot the parameter evolution
plt = plot()
for j in 1:p
    plot!(β[:,j]; label= L"\beta_{%$j}", c = colors[j], xlab ="t")
end
plt

# Set up time-varying regression models
mutable struct ParametersTvRegHetero
    σₑ::Float64
    Q::Vector{PDMat{Float64}}
    μ₀::Vector{Float64}
    Σ₀::PDMat{Float64}
    y::Vector{Float64}
    Z::Matrix{Float64}
end 

# Run FFBS to get a proposal for t = 1
Σₑ = σₑ^2
A = collect(I(p))
C = zeros(1,p,T)
for t in 1:T
    C[:,:,t] = Z[t,:]
end
B = 0.0
U = zeros(T,1)
Σₙ = Q # Heteroscedastic
FFBSinitdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, 500);
μₚ = mean(FFBSinitdraws[1,:,:], dims = 2)[:]
Σₚ = PDMat(cov(FFBSinitdraws[1,:,:], dims = 2))

initproposal(θ) = MvNormal(μₚ, Σₚ) # This is the proposal for θ₁ 
transition(θ, state, t) = MvNormal(state, θ.Q[t])
observation(θ, state, t) = Normal(θ.Z[t,:] ⋅ state, θ.σₑ)
prior(θ) = MvNormal(θ.μ₀, θ.Σ₀)
θ = ParametersTvRegHetero(σₑ, Q, μ₀, Σ₀, y, Z)

# Algorithm settings
Nₚ = 20       # Number of particles
Nₛ = 1000     # Number of samples from posterior

# PGAS with initial proposal = prior
println("Generating $Nₛ PGAS draws based on $Nₚ particles - prior as initial proposal")
@time PGASdraws_prior = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation); 
PGASmean_prior = mean(PGASdraws_prior, dims = 3)[:,:,1]
PGASquantiles_prior = myquantile(PGASdraws_prior, [0.025, 0.975], dims = 3)

println("Generating $Nₛ PGAS draws based on $Nₚ particles - FFBS as initial proposal")
@time PGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation,        
    initproposal); 
PGASmean = mean(PGASdraws, dims = 3)[:,:,1]
PGASquantiles = myquantile(PGASdraws, [0.025, 0.975], dims = 3)

# Plot update rates
update_rate = sum(abs.(diff(PGASdraws[:,1,:]; dims = 2)) .> 0; dims=2) / Nₛ
plt1 = plot(update_rate; label=false, ylim=[0, 1], legend=:bottomleft, xlabel="Iteration",
    ylabel="Update rate", title = "PGAS with FFBS init proposal")
hline!([1 - 1 / Nₚ]; label="Optimal rate for N: $(Nₚ) particles", c = colors[3], 
    legend = :bottomright, lw = 1, linestyle = :dash)

update_rate = sum(abs.(diff(PGASdraws_prior[:,1,:]; dims = 2)) .> 0; dims=2) / Nₛ
plt2 = plot(update_rate; label=false, ylim=[0, 1], legend=:bottomleft, xlabel="Iteration",
    ylabel="Update rate", title = "PGAS with prior init proposal")
hline!([1 - 1 / Nₚ]; label="Optimal rate for N: $(Nₚ) particles", c = colors[3], 
    legend = :bottomright, lw = 1, linestyle = :dash)

plot(plt1, plt2, layout = (1,2), size = (800,300), margins = 5mm)

## FFBS
Σₑ = σₑ^2
A = collect(I(p))
C = zeros(1,p,T)
for t in 1:T
    C[:,:,t] = Z[t,:]
end
B = 0.0
U = zeros(T,1)
Σₙ = Q # Heteroscedastic

println("FFBS")
@time FFBSdraws = FFBS(U, y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, Nₛ);
FFBSmean = mean(FFBSdraws, dims = 3)[2:end,:,1] # Exclude initial state at t=0
FFBSquantiles = myquantile(FFBSdraws, [0.025, 0.975], dims = 3)[2:end,:,:]

# FFBS as initial proposal
plottrue = false
p = length(prior(θ))
plt = []
for j in 1:p
    # True state evolution
    if plottrue
        plt_tmp = plot(x, c = :gray, lw = 1, title = L"\beta_{%$(j-1)}", 
            label = "true state", legend = :bottom)
    else
        plt_tmp = plot(title = L"\beta_{%$(j-1)}", legend = :bottom)
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


# Prior as initial proposal
plottrue = false
p = length(prior(θ))
plt = []
for j in 1:p
    # True state evolution
    if plottrue
        plt_tmp = plot(x, c = :gray, lw = 1, title = L"\beta_{%$(j-1)}", 
            label = "true state", legend = :bottom)
    else
        plt_tmp = plot(title = L"\beta_{%$(j-1)}", legend = :bottom)
    end
    # PGAS
    plot!(PGASmean_prior[:,j], lw = 1,
        c = colors[j], linestyle = :solid, label = "PGAS(N=$Nₚ)")
    plot!(PGASquantiles_prior[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[j], linecolor = colors[j],
        label = "", lw = 0) 
    # FFBS
    plot!(FFBSmean[:,j]; color = :black, lw = 1, linestyle = :dash, label="FFBS")
    plot!(FFBSquantiles[:,j,1], lw = 1, c = :black, linestyle = :dash, label="")
    plot!(FFBSquantiles[:,j,2], lw = 1, c = :black, linestyle = :dash, label = "")

    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,p), size = (800,300))
