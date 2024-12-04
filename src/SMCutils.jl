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