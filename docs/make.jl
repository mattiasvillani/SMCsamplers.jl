using SMCsamplers
using Documenter

DocMeta.setdocmeta!(SMCsamplers, :DocTestSetup, :(using SMCsamplers); recursive=true)

makedocs(;
    modules=[SMCsamplers],
    authors="Mattias Villani",
    sitename="SMCsamplers.jl",
    format=Documenter.HTML(;
        canonical="https://mattiasvillani.github.io/SMCsamplers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mattiasvillani/SMCsamplers.jl",
    devbranch="main",
)
