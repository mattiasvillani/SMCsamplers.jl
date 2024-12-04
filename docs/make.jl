using SMCsamplers
using Documenter

DocMeta.setdocmeta!(SMCsamplers, :DocTestSetup, :(using SMCsamplers); recursive=true)

makedocs(;
    sitename="SMCsamplers.jl",
    authors="Mattias Villani",
    format = Documenter.HTML(prettyurls = false),
    doctest = true,
    pages = Any[
        "Home" => "index.md",
        "Particle Gibbs" => "ParticleGibbs.md",
        "FFBS" => "FFBS.md",
    ],
)

#deploydocs(;
#    repo="github.com/mattiasvillani/SMCsamplers.jl",
#  devbranch="main",
#)
