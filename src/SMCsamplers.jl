module SMCsamplers

using Plots, LinearAlgebra, Distributions, Statistics, Random
using Measures, PDMats, LaTeXStrings, ForwardDiff, LineSearches

include("Misc.jl")
export NewtonRaphson

include("KalmanFilters.jl")
export kalmanfilter_update, kalmanfilter_update_extended, kalmanfilter_update_unscented
export kalmanfilter_update_extended_iter, kalmanfilter_update_extended_iter_line

include("FFBSsamplers.jl")
export FFBS, FFBSx, FFBS_unscented, FFBS_laplace

include("ParticleGibbsSamplers.jl")
export PGASupdate, PGASsampler

include("SMCutils.jl")
export quantile_multidim, systematic, multinomial, ESS # TODO: add stratified resampling

end
  