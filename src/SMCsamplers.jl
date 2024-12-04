module SMCsamplers

colors = [
    "#6C8EBF", "#c0a34d", "#780000", "#007878",     
    "#b5c6df","#eadaaa","#AE6666", "#4CA0A0","#bf9d6c", "#3A6B35", 
    "#9d6a6d","#d9c6c7", "#98bbb9", "#bf8d6c", 
    "#CBD18F"]

export colors

using Plots, LinearAlgebra, Distributions, Statistics, Random
using Measures, PDMats, LaTeXStrings

include("FFBSsamplers.jl")
export FFBS, FFBSx, FFBS_unscented

include("ParticleGibbsSamplers.jl")
export PGASupdate, PGASsampler

include("SMCutils.jl")
export systematic, multinomial, ESS

end
