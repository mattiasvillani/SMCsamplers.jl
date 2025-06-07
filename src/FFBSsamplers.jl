""" 
    kalmanfilter_update(μ, Ω, u, y, A, B, C, Σₑ, Σₙ) 

A single Kalman filter update at time t of the state space model: 

yₜ = Cxₜ + εₜ,           εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where
xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.

"""
function kalmanfilter_update(μ, Ω, u, y, A, B, C, Σₑ, Σₙ)

    # Prior propagation step - moving state forward without new measurement
    μ̄ = A*μ .+ B*u
    Ω̄ = A*Ω*A' + Σₙ

    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    K = Ω̄*C' / (C*Ω̄*C' .+ Σₑ) # Kalman Gain
    μ = μ̄ + K*(y .- C*μ̄)
    Ω = (I - K*C)*Ω̄
    return μ, Ω, μ̄, Ω̄

end

# Backward sampling step for a given t. A is a vector or Matrix here.
function BackwardSim(x, μ_filt, Σ_filt, μ_pred, Σ_pred, A::AbstractArray, t)
    μback = μ_filt + Σ_filt*A'*( Σ_pred\(x .- μ_pred) )
    Σback = Σ_filt -  Σ_filt*A'*( Σ_pred \ A )*Σ_filt
    try 
        return rand(MvNormal(μback, Hermitian(Σback)))
    catch # Σₙ ≤ eps() so, Σ_pred ≈ Σ_filt and Σback ≈ 0. xₜ = xₜ₊₁ 
        return x   
    end
end

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


""" 
    FFBS(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀) 

Forward filtering and backward sampling a *single* draw from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the state space model:

yₜ = Cxₜ + εₜ,           εₜ ~ N(0,Σₑ)         Measurement equation

xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where

xₜ is the n-dim state

uₜ is the m-dim control

yₜ is the k-dim observed data. 

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀.
A, C, Σₑ and Σₙ can be deterministically time-varying by passing 3D arrays of size n×n×T.

Note: This is generating a *single* draw from the joint smoothing posterior. Typically used as a step in a Gibbs sampler.

""" 
function FFBS(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀)

    T = size(Y,1)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticC = (ndims(C) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
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
        Ct = staticC ? C : @view C[:,:,t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update(μ, Σ, u, y, At, B, Ct, Σₑt, Σₙt)
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
    x0 = BackwardSim(X[1,:], μ₀, Matrix(Σ₀), μ_pred[1,:], Σ_pred[:,:,1], A, 0)

    return [x0'; X]

end

function FFBS(U, Y, A, B, C, Σₑ, Σₙ, μ₀, Σ₀, nIter)

    T = size(Y,1)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticC = (ndims(C) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
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
        Ct = staticC ? C : @view C[:,:,t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update(μ, Σ, u, y, At, B, Ct, Σₑt, Σₙt)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    Xdraws = zeros(T+1, n, nIter)  
    for i = 1:nIter 

        # Backward sampling for t = T, T-1, ..., 1
        X = zeros(T, n) 
        X[T,:] = rand(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])))
        for t = (T-1):-1:1
            X[t,:] = BackwardSim(X[t+1,:], μ_filter[t,:], Σ_filter[:,:,t], μ_pred[t+1,:],   
                Σ_pred[:,:,t+1], A, t)
        end

        # Finally, sample state at t = 0
        x0 = BackwardSim(X[1,:], μ₀, Matrix(Σ₀), μ_pred[1,:], Σ_pred[:,:,1], A, 0)

        Xdraws[:,:,i] = [x0'; X]

    end

    return Xdraws

end



""" 
    kalmanfilter_update_extended(μ, Ω, u, y, A, B, C, ∂C, Cargs, Σₑ, Σₙ) 

A single extended Kalman filter update at time t of the state space model: 

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where C(xₜ) is a non-linear measurement function

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.

"""
function kalmanfilter_update_extended(μ, Ω, u, y, A, B, C::Function, ∂C::Function, Cargs, Σₑ, Σₙ)

    # Prior propagation step
    μ̄ = A*μ + B*u
    Ω̄ = A*Ω*A' + Σₙ
    C̄ = ∂C(μ̄[:], Cargs) # y_t = C(θ_t) + ε_t

    # Measurement update
    K = Ω̄*C̄' / (C̄*Ω̄*C̄' .+ Σₑ) 
    μ = μ̄ + K*(y .- C(μ̄[:], Cargs))
    Ω = (I - K*C̄)*Ω̄
    
    return μ, Ω, μ̄, Ω̄

end

""" 
    FFBSx(U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀) 

Forward filtering and backward sampling a *single* draw from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the state space model with nonlinear measurement equation:

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation

xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where

C(xₜ) is a non-linear function that we can ForwardDiff.jl to get the Jacobian

xₜ is the n-dim state

uₜ is the m-dim control

yₜ is the k-dim observed data. 

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀

""" 
function FFBSx(U, Y, A, B, C, ∂C, Cargs, Σₑ, Σₙ, μ₀, Σ₀)

    T = size(Y,1)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true
    staticCargs = (ndims(Cargs) == 3 || eltype(Cargs) <: Vector) ? false : true

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Cargs_t = staticCargs ? Cargs : Cargs[t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update_extended(μ, Σ, u, y, At, B, C, ∂C, Cargs_t, Σₑt, Σₙt)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    # Backward sampling for t = T, T-1, ..., 1
    X = zeros(T, n) 
    X[T,:] = rand(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])))
    for t = (T-1):-1:1
        X[t,:]= BackwardSim(X[t+1,:], μ_filter[t,:], Σ_filter[:,:,t], μ_pred[t+1,:], 
            Σ_pred[:,:,t+1], A, t)
    end

    # Finally, sample state at t = 0
    x0 = BackwardSim(X[1,:], μ₀, Σ₀, μ_pred[1,:], Σ_pred[:,:,1], A, 0)

    return [x0'; X]

end


""" 
    kalmanfilter_update_unscented(μ, Ω, u, y, A, B, C, Cargs, Σₑ, Σₙ) 

A single unscented Kalman filter update at time t of the state space model: 

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation
xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where C(xₜ) is a non-linear measurement function

xₜ is the n-dim state
uₜ is the m-dim control
yₜ is the k-dim observed data. 

Reference: Thrun, Burgard and Fox (2006). Probabilistic Robotics, Algorithm Kalman_filter in Table 3.4

"""
function kalmanfilter_update_unscented(μ, Ω, u, y, A, B, C::Function, Cargs, Σₑ, Σₙ, 
        γ, ωₘ, ωₛ)

    # Prediction step - moving state forward without new measurement
    μ̄ = A*μ + B*u
    Ω̄ = A*Ω*A' + Σₙ
   
    L̄ = cholesky(Ω̄).L # FIXME: This can be time-consuming, compute more efficiently?
    
    # Measurement update - updating the N(μ̄, Ω̄) prior with the new data point
    X̄ = [μ̄ (μ̄ .+ γ*L̄) (μ̄ .- γ*L̄)] # n×(2n+1) matrix with sigma points approx N(μ̄, Ω̄)
    Ȳ = mapslices(x -> C(x, Cargs), X̄, dims = 1) # Apply the C() function to each column
    ŷ = sum(Ȳ .* ωₘ', dims = 2)
    S = sum((Ȳ .- ŷ)*(Ȳ .- ŷ)' .* ωₛ', dims = 2) + Σₑ
    Ω̄xy = sum((X̄ .- μ̄)*(Ȳ .- ŷ)' .* ωₛ', dims = 2)
    K = Ω̄xy / S
    μ = μ̄ + K*(y .- ŷ)
    Ω = Ω̄ - K*S*K'

    return μ, Ω, μ̄, Ω̄

end


""" 
    FFBS_unscented(U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀) 

Forward filtering and backward sampling a *single* draw from the joint smoothing posterior 
p(x1,...xT | y1,...,yT) of the state space model with nonlinear measurement equation:

yₜ = C(xₜ) + εₜ,         εₜ ~ N(0,Σₑ)         Measurement equation

xₜ = Axₜ₋₁+ Buₜ + ηₜ,    ηₜ ~ N(0,Σₙ)         State equation

where

C(xₜ) is a non-linear function

xₜ is the n-dim state

uₜ is the m-dim control

yₜ is the k-dim observed data. 

The observed data observations are the rows of the T×k matrix Y
The control signals are the rows of the T×m matrix U
μ₀ and Σ₀ are the mean and covariance of the initial state vector x₀


""" 
function FFBS_unscented(U, Y, A, B, C, Cargs, Σₑ, Σₙ, μ₀, Σ₀; α = 1, β = 0, κ = 1)

    T = size(Y,1)   # Number of time steps
    n = length(μ₀)  # Dimension of the state vector  
    r = size(Y,2)   # Dimension of the observed data vector
    q = size(U,2)   # Dimension of the control vector
    staticA = (ndims(A) == 3) ? false : true
    staticΣₑ = (ndims(Σₑ) == 3  || eltype(Σₑ) <: PDMat) ? false : true
    staticΣₙ = (ndims(Σₙ) == 3  || eltype(Σₙ) <: PDMat) ? false : true

    # Set up the weights for the unscented Kalman filter
    λ = α^2*(n + κ) - n # λ = 3-n # me = 1
    ωₘ = [λ/(n + λ); ones(2*n)/(2*(n + λ))]
    ωₛ = [λ/(n + λ) + (1 - α^2 + β); ωₘ[2:end]]
    γ = sqrt(n + λ) # Ganna: sqrt(3) # sqrt(n + 1)

    # Run Kalman filter and collect matrices
    μ_filter = zeros(T, n)      # Storage of μₜₜ
    Σ_filter = zeros(n, n, T)   # Storage of Σₜₜ
    μ_pred = zeros(T, n)        # Storage of μₜ,ₜ₋₁
    Σ_pred = zeros(n, n, T)     # Storage of Σₜ,ₜ₋₁

    μ = deepcopy(μ₀)
    Σ = deepcopy(Σ₀)
    for t = 1:T
        At = staticA ? A : @view A[:,:,t]
        Cargs_t = Cargs[t]
        Σₑt = staticΣₑ ? Σₑ : Σₑ[t]
        Σₙt = staticΣₙ ? Σₙ : Σₙ[t]
        u = (q == 1) ? U[t] : U[t,:]
        y = (r == 1) ? Y[t] : Y[t,:]
        μ, Σ, μ̄, Σ̄ = kalmanfilter_update_unscented(μ, Σ, u, y, At, B, C, Cargs_t, 
            Σₑt, Σₙt, γ, ωₘ, ωₛ)
        μ_filter[t,:] .= μ
        Σ_filter[:,:,t] .= Σ
        μ_pred[t,:] .= μ̄
        Σ_pred[:,:,t] .= Σ̄
    end

    # Backward sampling for t = T, T-1, ..., 1
    X = zeros(T, n) 
    X[T,:] = rand(MvNormal(μ_filter[T,:], Hermitian(Σ_filter[:,:,T])))
    for t = (T-1):-1:1
        X[t,:]= BackwardSim(X[t+1,:], μ_filter[t,:], Σ_filter[:,:,t], μ_pred[t+1,:], 
            Σ_pred[:,:,t+1], A, t)
    end

    # Finally, sample state at t = 0
    x0 = BackwardSim(X[1,:], μ₀, Σ₀, μ_pred[1,:], Σ_pred[:,:,1], A, 0)

    return [x0'; X]

end