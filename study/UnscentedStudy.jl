using SMCsamplers, Plots, Distributions, LaTeXStrings, Random
using LinearAlgebra, Measures
using Utils: CovMatEquiCorr, Cov2Corr

figFolder = joinpath(dirname(@__DIR__), "study")


# Plot settings
gr(legend = :topleft, grid = false, color = colors[2], lw = 2, legendfontsize= 10,
    xtickfontsize=10, ytickfontsize=10, xguidefontsize=14, yguidefontsize=14,
    titlefontsize = 16, markerstrokecolor = :auto)

α = 1; β = 0; κ = 1;
n = 2

λ = α^2*(n + κ) - n
γ = sqrt(n + λ)

pBlock = [2]

plt = []
settings = [Dict(:σₓ => [1,1], :ρ => [0.5]) 
            Dict(:σₓ => [1,1], :ρ => [0.9]) 
            Dict(:σₓ => [1,5], :ρ => [0.9])
            ]

for j in 1:length(settings)
    σₓ = settings[j][:σₓ]
    ρ = settings[j][:ρ]

    Ω̄ = CovMatEquiCorr(σₓ, ρ, pBlock)
    μ̄ = 1*ones(n)
    L̄₁ = cholesky(Ω̄).L;
    L̄₂ = sqrt(Ω̄)


    ωₘ = [λ/(n + λ); ones(2*n)/(2*(n + λ))]
    ωₛ = [λ/(n + λ) + (1 - α^2 + β); ωₘ[2:end]]

    chisqVals = quantile.(Chisq(length(μ̄)), reverse([0.1 0.25 0.5 0.75 0.95]))
    pdfAtContours = (1/√det(2*π*Ω̄))*exp.(-0.5*chisqVals)

    X̄₁ = [μ̄ (μ̄ .+ γ*L̄₁) (μ̄ .- γ*L̄₁)]
    X̄₂ = [μ̄ (μ̄ .+ γ*L̄₂) (μ̄ .- γ*L̄₂)]

    # Plot bivariate normal contours and sigma points
    σ = sqrt.(diag(Ω̄))
    x1grid = range(μ̄[1] - 3*σ[1], μ̄[1] + 3*σ[1], length = 100)
    x2grid = range(μ̄[2] - 3*σ[2], μ̄[2] + 3*σ[2], length = 100)
    p = contour(size=(450,450), x1grid, x2grid, (x1,x2) -> pdf(MvNormal(μ̄, Ω̄), [x1,x2]), 
        levels = pdfAtContours[:], fill = true, color = :Blues, 
        xlabel = L"x_1", ylabel = L"x_2", lw = 1, linecolor = :black, 
        colorbar = false,
        margin = 5mm, title = L"\sigma_1 = %$(σ[1]), \sigma_2 = %$(σ[2]), ρ = %$(ρ[1])")
    scatter!(X̄₁[1,:], X̄₁[2,:], markersize = 8*ωₛ/maximum(ωₛ), 
        markerstrokecolor = colors[3], markercolor = colors[3], 
        label = "σ-points (cholesky)")
    scatter!(X̄₂[1,:], X̄₂[2,:], markersize = 8*ωₛ/maximum(ωₛ), 
        markerstrokecolor = colors[2], markercolor = colors[2], 
        label = "σ-points (matrix sqrt)")
    push!(plt, p)

end
plot(plt..., layout = (1,3), size = (1200, 500), margin = 5mm)
savefig(figFolder*"/SigmaPoints_beta$β.pdf")