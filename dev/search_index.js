var documenterSearchIndex = {"docs":
[{"location":"home/","page":"Home","title":"Home","text":"CurrentModule = SMCsamplers","category":"page"},{"location":"home/#SMCsamplers.jl","page":"Home","title":"SMCsamplers.jl","text":"","category":"section"},{"location":"home/#Description","page":"Home","title":"Description","text":"","category":"section"},{"location":"home/","page":"Home","title":"Home","text":"This is the start of a package for some Sequential Monte Carlo samplers in Julia. Some examples scripts can be found in the examples folder.","category":"page"},{"location":"home/#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"home/","page":"Home","title":"Home","text":"Install from the Julia package manager (via Github) by typing ] in the Julia REPL:","category":"page"},{"location":"home/","page":"Home","title":"Home","text":"] add git@github.com:mattiasvillani/SMCsamplers.jl.git","category":"page"},{"location":"home/#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"home/","page":"Home","title":"Home","text":"# PGAS to simulate from the posterior of the state in stochastic volatility (SV) model:\n#   x₀ ∼ N(0,σ₀)\n#   xₜ = a⋅xₜ₋₁ + νₜ, νₜ ∼ N(0,σᵥ)\n#   yₜ = exp(xₜ/2)εₜ, εₜ ∼ N(0,1)\n\nusing SMCsamplers, Plots, Distributions, LaTeXStrings, Random\n\n# Set up SV model structure for PGAS\nmutable struct SVParams \n    a::Float64\n    σᵥ::Float64\n    σ₀::Float64\nend\nprior(θ) = Normal(0, θ.σ₀)\ntransition(θ, state, t) = Normal(θ.a * state, θ.σᵥ)  \nobservation(θ, state, t) = Normal(0, exp(state/2))\n\n# Set model parameters\na = 0.9         # Persistence\nσᵥ = 1          # State std deviation\nσ₀ = 0.5        # Initial observation std deviation\nT = 200         # Length of time series\n\nθ = SVParams(a, σᵥ, σₑ) # Set up parameter struct for PGAS\n\n# Algorithm settings\nNₚ = 20         # Number of particles for PGAS\nNₛ = 1000       # Number of samples from posterior\n\n# Simulate data from SV model\nx = zeros(T)\ny = zeros(T)\nx0 = rand(prior(θ))\nfor t in 1:T\n    if t == 1\n        x[t] = rand(transition(θ, x0, t))\n    else\n        x[t] = rand(transition(θ, x[t-1], t))\n    end\n    y[t] = rand(observation(θ, x[t], t))\nend \n\n# Simulate from joint smoothing posterior using PGAS for given static parameters\nPGASdraws = PGASsampler(y, θ, Nₛ, Nₚ, prior, transition, observation) # returns (T, 1, Nₛ) array\n","category":"page"},{"location":"FFBS/","page":"FFBS","title":"FFBS","text":"FFBS\nFFBSx\nFFBS_unscented","category":"page"},{"location":"FFBS/#SMCsamplers.FFBS","page":"FFBS","title":"SMCsamplers.FFBS","text":"FFBS(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀)\n\nForward filtering and backward sampling a single draw from the joint smoothing posterior  p(x1,...xT | y1,...,yT) of the state space model:\n\nyₜ = Cxₜ + εₜ,           εₜ ~ N(0,Σₑ)         Measurement equation\n\nxₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation\n\nwhere\n\nxₜ is the n-dim state\n\nuₜ is the m-dim control\n\nyₜ is the k-dim observed data. \n\nThe observed data observations are the rows of the T×k matrix Y The control signals are the rows of the T×m matrix U μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀. A, C, Σₑ and Σₙ can be deterministically time-varying by passing 3D arrays of size n×n×T.\n\nNote: This is generating a single draw from the joint smoothing posterior. Typically used as a step in a Gibbs sampler.\n\n\n\n\n\n","category":"function"},{"location":"FFBS/#SMCsamplers.FFBSx","page":"FFBS","title":"SMCsamplers.FFBSx","text":"FFBSx(U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀)\n\nForward filtering and backward sampling a single draw from the joint smoothing posterior  p(x1,...xT | y1,...,yT) of the state space model with nonlinear measurement equation:\n\nyₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation\n\nxₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation\n\nwhere\n\nC(xₜ) is a non-linear function that we can ForwardDiff.jl to get the Jacobian\n\nxₜ is the n-dim state\n\nuₜ is the m-dim control\n\nyₜ is the k-dim observed data. \n\nThe observed data observations are the rows of the T×k matrix Y The control signals are the rows of the T×m matrix U μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀\n\n\n\n\n\n","category":"function"},{"location":"FFBS/#SMCsamplers.FFBS_unscented","page":"FFBS","title":"SMCsamplers.FFBS_unscented","text":"FFBS_unscented(U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀)\n\nForward filtering and backward sampling a single draw from the joint smoothing posterior  p(x1,...xT | y1,...,yT) of the state space model with nonlinear measurement equation:\n\nyₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation\n\nxₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation\n\nwhere\n\nC(xₜ) is a non-linear function\n\nxₜ is the n-dim state\n\nuₜ is the m-dim control\n\nyₜ is the k-dim observed data. \n\nThe observed data observations are the rows of the T×k matrix Y The control signals are the rows of the T×m matrix U μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀\n\n\n\n\n\n","category":"function"},{"location":"KalmanFilters/","page":"-","title":"-","text":"kalmanfilter_update\nkalmanfilter_update_extended\nkalmanfilter_update_unscented","category":"page"},{"location":"KalmanFilters/#SMCsamplers.kalmanfilter_update","page":"-","title":"SMCsamplers.kalmanfilter_update","text":"kalmanfilter_update(μ, Ω, u, y, A, B, C, Σₑ, Σₙ)\n\nA single Kalman filter update at time t of the state space model: \n\nyₜ = Cxₜ + εₜ,           εₜ ~ N(0,Σₑ)         Measurement equation xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation\n\nwhere xₜ is the n-dim state uₜ is the m-dim control yₜ is the k-dim observed data. \n\nReference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.\n\n\n\n\n\n","category":"function"},{"location":"KalmanFilters/#SMCsamplers.kalmanfilter_update_extended","page":"-","title":"SMCsamplers.kalmanfilter_update_extended","text":"kalmanfilter_update_extended(μ, Ω, u, y, A, B, C, ∂C, Cargs, Σₑ, Σₙ)\n\nA single extended Kalman filter update at time t of the state space model: \n\nyₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation\n\nwhere C(xₜ) is a non-linear measurement function\n\nxₜ is the n-dim state uₜ is the m-dim control yₜ is the k-dim observed data. \n\nReference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.\n\n\n\n\n\n","category":"function"},{"location":"KalmanFilters/#SMCsamplers.kalmanfilter_update_unscented","page":"-","title":"SMCsamplers.kalmanfilter_update_unscented","text":"kalmanfilter_update_unscented(μ, Ω, u, y, A, B, C, Cargs, Σₑ, Σₙ)\n\nA single unscented Kalman filter update at time t of the state space model: \n\nyₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation\n\nwhere C(xₜ) is a non-linear measurement function\n\nxₜ is the n-dim state uₜ is the m-dim control yₜ is the k-dim observed data. \n\nReference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.4\n\n\n\n\n\n","category":"function"},{"location":"ParticleGibbs/","page":"Particle Gibbs","title":"Particle Gibbs","text":"PGASsampler\nPGASupdate","category":"page"},{"location":"ParticleGibbs/#SMCsamplers.PGASsampler","page":"Particle Gibbs","title":"SMCsamplers.PGASsampler","text":"PGASsampler(y, θ, nDraws, N, prior, transition, observation, \n    initproposal = prior, resampler = systematic)\n\nUses the PGAS algorithm with N particles to simulate nDraws from the joint smoothing posterior of the state xₜ in the state space model determined by the three functions (that all return distributions):\n\nprior is the prior for the initial state p(x₁) \ntransition is the transition density p(xₜ | xₜ₋₁)\nobservation is the observation density p(yₜ | xₜ)\n\nθ is a struct with all the parameters in the model needed to evaluate the prior, transition and observation densities.\n\ninitproposal is the proposal distribution for x₁ and resampler is the resampling function which defaults to systematic resampling.\n\n\n\n\n\n","category":"function"},{"location":"ParticleGibbs/#SMCsamplers.PGASupdate","page":"Particle Gibbs","title":"SMCsamplers.PGASupdate","text":"PGASupdate(y, p, N, θ, prior, transition, observation, \n    initproposal, resampler, Xref = nothing)\n\nSingle update step of the PGAS algorithm with N particles to simulate from the joint smoothing posterior of the state xₜ in the state space model determined by the three functions (that all return distributions):\n\nprior is the prior for the initial state p(x₁) \ntransition is the transition density p(xₜ | xₜ₋₁)\nobservation is the observation density p(yₜ | xₜ)\n\nθ is a struct with all the parameters in the model needed to evaluate the prior, transition and observation densities.\n\ninitproposal is the proposal distribution for x₁ and resampler is the resampling function which defaults to systematic resampling.\n\nXref is a p×T matrix with conditioning particle path - if nothing, unconditional PF is run\n\n\n\n\n\n","category":"function"},{"location":"#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"","page":"Index","title":"Index","text":"","category":"page"}]
}
