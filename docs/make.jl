using SMCsamplers
using Documenter

DocMeta.setdocmeta!(SMCsamplers, :DocTestSetup, :(using SMCsamplers); recursive=true)

makedocs(;
    sitename="SMCsamplers.jl",
    modules=[SMCsamplers],
    authors="Mattias Villani",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mattiasvillani.github.io/SMCsamplers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = Any[
        "Home" => "home.md",
        "Particle Gibbs" => "ParticleGibbs.md",
        "FFBS" => "FFBS.md",
        "Index" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattiasvillani/SMCsamplers.jl",
)
