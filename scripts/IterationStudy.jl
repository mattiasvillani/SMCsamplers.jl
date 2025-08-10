# ### First some preliminaries:
using SMCsamplers, Plots, Distributions, LaTeXStrings, Random, ForwardDiff, PDMats
using LinearAlgebra, Measures

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

# ### Simulate data from a SAR(1,1) model with s = 12 
s = 12
p = 1
P = 1
nState = p + P              
T = 200
time = 1:T
s = 12
T₊ = 2*(s+1)+T # include 2*(s+1) presample values
ϕ(t) = 0.99*sin(2π*t/T) # Time-varying AR coefficient
Φ(t) = (t <= T/2 + 2*(s+1)) ? -0.5 : 0.95
σₑ = 0.2
y = zeros(T₊)
paramEvol = zeros(T₊, 2)
for t = (s+2):(T₊)
    y[t] = ϕ(t)*y[t-1] + Φ(t)*y[t-s] - ϕ(t)*Φ(t)*y[t-s-1] + σₑ*randn() 
    paramEvol[t,:] .= ϕ(t), Φ(t)
end
y = y[(s+2):end] # Remove presample values
paramEvol = paramEvol[(s+2):end,:];

# ### Set up the SAR model as a nonlinear regression
lag1 = [NaN;y[1:end-1]] # Lag 1
lagS = [NaN*ones(s);y[1:end-s]] # Lag s
lagSplus1 = [NaN*ones(s+1);y[1:(end-s-1)]] # Lag s+1

Z = [lag1 lagS lagSplus1]
Z = Z[s+2:end, :]           # Remove first s rows with NaNs
y = y[s+2:end]              # Remove lost observations
paramEvol = paramEvol[s+2:end,:];
 
# ### Plot the data and the time-varying parameters
plot(time, y, label = "data", lw = 2, c = colors[1], xlabel = "time", ylabel = "", 
    ylims = (-1,1), legend = :bottomleft)
plot!(time, paramEvol[:,1], label = L"\phi_t", lw = 2, c = colors[2])
plot!(time, paramEvol[:,2], label = L"\Phi_t", lw = 2, c = colors[3])

# ### Set up SAR model structure for PGAS and set static parameter values
mutable struct SARParams 
    σₑ::Float64
    σᵥ::Float64
    σ₀::Float64
    Z::Matrix{Float64}
end

prior(θ) = MvNormal(zeros(p+P), θ.σ₀)
transition(θ, state, t) = MvNormal(state, θ.σᵥ)  
observation(θ, state, t) = Normal(θ.Z[t,:]' ⋅ [state[1];state[2];-state[1]*state[2]], θ.σₑ); 

# Set up static parameter values
σₑ = σₑ                 # Noise std deviation from static model
σᵥ = 0.1                # State std deviation
σ₀ = 1                  # Initial state std deviation
θ = SARParams(σₑ, σᵥ, σ₀, Z)

nSim = 1000;            # Number of samples from posterior

# ### PGAS sampling
nParticles = 100         # Number of particles for PGAS
PGASdraws = PGASsampler(y, θ, nSim, nParticles, prior, transition, 
    observation);
PGASmean = mean(PGASdraws, dims = 3)[:,:,1];
PGASquantiles = myquantile(PGASdraws, [0.025, 0.975], dims = 3);
    

plt = [];
titles = [L"\phi_{t}",L"\Phi_{t}"];
legendPos = [:bottomleft, :bottomright];
for j = 1:nState
    #plt_tmp = plot(time, paramEvol[:,j], lw = 2, c = :black, linestyle = :dash, 
    #     label = "true", title = titles[j], legend = legendPos[j])
    plt_tmp = plot([0;time], EKFmean[:,j], lw = 1.5, c = colors[1], linestyle = :solid, 
        label = "EKF(1)", title = titles[j], legend = legendPos[j])
    plot!([0;time], EKFquantiles[:,j,1], fillrange = EKFquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[1], linecolor = colors[1], label = "", lw = 0) 
    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,2), size = (800, 300), xlabel = "time", bottommargin = 5mm)

plt = [];
titles = [L"\phi_{t}",L"\Phi_{t}"];
legendPos = [:bottomleft, :bottomright];
for j = 1:nState
    #plt_tmp = plot(time, paramEvol[:,j], lw = 2, c = :black, linestyle = :dash, 
    #    label = "true", title = titles[j], legend = legendPos[j])
    plt_tmp = plot(time, PGASmean[:,j], lw = 1, c = colors[1], linestyle = :solid, 
        label = "PGAS(N=$nParticles)", title = titles[j], legend = legendPos[j])
    plot!(time, PGASquantiles[:,j,1], fillrange = PGASquantiles[:,j,2],
        fillalpha = 0.2, fillcolor = colors[1], linecolor = colors[1], label = "", lw = 0) 
    push!(plt, plt_tmp)
end
plot(plt..., layout = (1,2), size = (800, 300), xlabel = "time", bottommargin = 5mm)

for (pltNumber, maxIter) in enumerate(1:3)
    IEKFdraws, μ_filterIEKF, Σ_filterIEKF = FFBSx(U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, 
        Σ₀, nSim, maxIter; filter_output = true);
    IEKFmean = mean(IEKFdraws, dims = 3)[:,:,1];
    IEKFquantiles = myquantile(IEKFdraws, [0.025, 0.975], dims = 3);
    for j = 1:nState
        plot!(plt[j], [0;time], IEKFmean[:,j], label = "EKF($(maxIter))", 
            color = colors[pltNumber+1], lw = 1.5)
        plot!(plt[j], [0;time], IEKFquantiles[:,j,1], lw = 1, c = colors[pltNumber+1], 
            linestyle = :solid, label = nothing)
        plot!(plt[j], [0;time], IEKFquantiles[:,j,2], lw = 1, c = colors[pltNumber+1], 
            linestyle = :solid, label = nothing)
    end
end
plot(plt..., layout = (1,2), size = (800, 300), xlabel = "time", bottommargin = 5mm)

