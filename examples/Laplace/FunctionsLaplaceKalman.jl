

using Distributions, Plots, ForwardDiff, LinearAlgebra, PDMats



# Laplace approx. (univariate only).
function laplace_approximation(logposterior, initial_guess, cov_scale=1.0)
    # Find mode (MAP estimate)
    function find_mode(x0)
        x = copy(x0)
        for _ in 1:100
        g = ForwardDiff.derivative(logposterior, x)
        H = ForwardDiff.derivative(x -> ForwardDiff.derivative(logposterior, x), x)
        Δx = -H \ g  # Newton-Raphson step
        x += Δx
        if norm(Δx) < 1e-6
            return x
        end
    end
    error("Mode finding did not converge")
    end
    
    θ_mode = find_mode(initial_guess)
    
    # Compute Hessian at mode
    Σ = -inv(ForwardDiff.derivative(θ_mode -> ForwardDiff.derivative(logposterior, θ_mode), θ_mode))  # Covariance matrix
    
    # Adjust covariance if needed (sometimes too narrow/wide)
    Σ *= cov_scale^2
    
    # Create normal approximation - can be removed if not needed.
    approx_posterior = Normal(θ_mode, sqrt(Σ))
      
    # Return results
    return (
        mode = θ_mode,
        covariance = Σ,
        distribution = approx_posterior
    )
end


# This function is identical to the on Mattias has in the packege.

# Backward sampling step for a given t. A is a scalar here.
function BackwardSim(x, μ_filt, Σ_filt, μ_pred, Σ_pred, A::Number, t)
    μback = μ_filt + Σ_filt*A'*( Σ_pred\(x .- μ_pred) )
    Σback = Σ_filt -  Σ_filt*A'*( Σ_pred \ [A] )*Σ_filt
    try 
        return rand(MvNormal(μback, Hermitian(Σback)))
    catch # Σₙ ≤ eps() so, Σ_pred ≈ Σ_filt and Σback ≈ 0. xₜ = xₜ₊₁ 
        return x   
    end
end




function laplace_kalmanfilter_update(μ, Ω, u, y, A, B, logLik, Σₙ)

    # Prior propagation step - moving state forward without new measurement
    μ̄ = A*μ .+ B*u
    Ω̄ = A*Ω*A' + Σₙ

    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    filt_logpost(x) = logLik(x, y) + logpdf(Normal(μ̄, sqrt(Ω̄ )), x)
    μ, Ω = laplace_approximation(filt_logpost, μ̄ )  # Initial guess 

    return μ, Ω, μ̄, Ω̄
end

# Basically the same as the package, but removes some stuff and added log likelihood as argument.
function laplace_FFBS(U, Y, A, B, Σₙ, μ₀, Σ₀, logLik)

    T = size(Y,1)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = laplace_kalmanfilter_update(μ, Σ, u, y, At, B, logLik, Σₙt)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    # Backward sampling for t = T, T-1, ..., 1
    X = zeros(T, n) 
    X[T,:] = rand(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])))
    for t = (T-1):-1:1
        X[t,:] = BackwardSim(X[t+1,:], μ_filter[t,:], Σ_filter[:,:,t], μ_pred[t+1,:],   
            Σ_pred[:,:,t+1], A, t)
    end

    # Finally, sample state at t = 0
    x0 = BackwardSim(X[1,:], [μ₀], [Σ₀;;], μ_pred[1,:], Σ_pred[:,:,1], A, 0)

    return [x0'; X]

end



#######################
# Laplace for multivariate:
function laplace_approximation_multi(logposterior, initial_guess; cov_scale=1.0)
    # Find mode (MAP estimate)
    function find_mode(x0)
        x = copy(x0)
        for _ in 1:100
            g = ForwardDiff.gradient(logposterior, x)
            H = ForwardDiff.hessian(logposterior, x)
            Δx = -H \ g  # Newton-Raphson step
            x += Δx
            if norm(Δx) < 1e-6
                return x
            end
        end
        error("Mode finding did not converge")
    end
    
    θ_mode = find_mode(initial_guess)
    
    # Compute Hessian at mode
    Σ = -inv(ForwardDiff.hessian(logposterior, θ_mode))  # Covariance matrix
    
    # Adjust covariance if needed (sometimes too narrow/wide).
    Σ *= cov_scale^2
    
    # Create multivariate normal approximation
    approx_posterior = MvNormal(θ_mode, Σ)
    
    
    # Return results
    return (
    mode = θ_mode,
    covariance = Σ,
    distribution = approx_posterior
    )
end