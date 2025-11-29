# Effective sample size
function ESS(w)
    return 1/sum(w.^2)
end

function multinomial(w)
    return rand(Categorical(w), length(w))
end

function systematic(w)
    
    m = length(w)
    u = rand()/m
    w_cum = cumsum(w)
    w_cum[end] = 1.0
    j = 1
    ind = zeros(Int, m)
    for i = 1:m
        while u > w_cum[j]
            j += 1
        end
        ind[i] = j
        u += 1/m
    end
    return ind
end

# Kullback-Leibler divergence for multivariate Gaussian distributions
function KLD(μ0, Σ0, μ1, Σ1)
    k = length(μ0)
    Δμ = μ1 - μ0
    invΣ1 = inv(Σ1)
    tr_term = tr(invΣ1 * Σ0)
    quad_term = Δμ' * invΣ1 * Δμ
    logdet0 = logdet(Σ0)[1]
    logdet1 = logdet(Σ1)[1]

    return 0.5*(tr_term + quad_term - k + logdet1 - logdet0)
end

# Helper function to make groups of equal size, last group may be smaller
function splitEqualGroups(y, Z, nPerGroup)
    nElements = length(y)
    nGroups = ceil(Int, nElements/nPerGroup)
    Y = []
    if !isempty(Z) && !isnothing(Z) 
        Zg = []
    else 
        Zg = nothing
    end
    i = 1
    while i <= nElements
        push!(Y, y[i:min(i+nPerGroup-1, nElements)])
        if !isempty(Z) && !isnothing(Z) 
            push!(Zg, Z[i:min(i+nPerGroup-1, nElements), :])
        end
        i += nPerGroup
    end
    return Y, Zg
end