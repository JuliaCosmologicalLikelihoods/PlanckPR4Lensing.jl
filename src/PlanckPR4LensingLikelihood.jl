module PlanckPR4LensingLikelihood

using LinearAlgebra, DelimitedFiles, Printf, Artifacts, NPZ

const DATA_DIR = artifact"planck_pr4_lensing_data"

function _dataset_path(filename)
    p = joinpath(DATA_DIR, "planck_pr4_data_complete", filename)
    isfile(p) || error("Data file not found: $p
Artifact may not be installed. Run `using Pkg; Pkg.instantiate()`.")
    return p
end

# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

struct LensingLikelihood
    cl_hat::Vector{Float64}
    cov_inv::Matrix{Float64}
    win_PP::Matrix{Float64}
    ell_PP::UnitRange{Int}
    delta_cl_fid::Vector{Float64}
    win_delta::Array{Float64,3}
    ell_delta::UnitRange{Int}
    lmin::Int
    lmax::Int
    nbins::Int
end

struct MargLensingLikelihood
    cl_hat::Vector{Float64}
    cov_inv::Matrix{Float64}
    win_PP::Matrix{Float64}
    ell_PP::UnitRange{Int}
    delta_cl_fid::Vector{Float64}
    win_delta::Array{Float64,3}
    ell_delta::UnitRange{Int}
    lmin::Int
    lmax::Int
    nbins::Int
end

# ------------------------------------------------------------------ #
#  Constructors                                                        #
# ------------------------------------------------------------------ #

function PlanckPR4Lensing()
    lmin, lmax, nbins = 2, 2500, 9
    
    cl_hat = npzread(_dataset_path("bandpowers.npy"))
    cov = npzread(_dataset_path("covmat.npy"))
    cov_inv = inv(cov)
    
    win_PP = npzread(_dataset_path("windows_PP.npy"))
    
    delta_cl_fid = npzread(_dataset_path("fid_correction.npy"))
    
    win_delta = permutedims(npzread(_dataset_path("win_delta.npy")), (2, 3, 1))
    
    return LensingLikelihood(vec(cl_hat), cov_inv, win_PP, lmin:lmax, vec(delta_cl_fid), win_delta, lmin:lmax, lmin, lmax, nbins)
end

function PlanckPR4LensingMarged()
    lmin, lmax, nbins = 2, 2500, 9
    
    cl_hat = npzread(_dataset_path("bandpowers.npy"))
    cov = npzread(_dataset_path("covmat_marged.npy"))
    cov_inv = inv(cov)
    
    win_PP = npzread(_dataset_path("windows_PP.npy"))

    delta_cl_fid = npzread(_dataset_path("fid_correction.npy"))

    win_delta = permutedims(npzread(_dataset_path("win_delta.npy")), (2, 3, 1))
    
    return MargLensingLikelihood(vec(cl_hat), cov_inv, win_PP, lmin:lmax, vec(delta_cl_fid), win_delta, lmin:lmax, lmin, lmax, nbins)
end

# ... rest of file is the same
function compute_total_binned(lik::LensingLikelihood, cl_pp, cl_tt, cl_ee, cl_te; A_planck=1.0)
    C_b = lik.win_PP * cl_pp[lik.ell_PP .+ 1]
    cl_tt_scaled = cl_tt[lik.ell_delta .+ 1] ./ A_planck^2
    cl_ee_scaled = cl_ee[lik.ell_delta .+ 1] ./ A_planck^2
    cl_te_scaled = cl_te[lik.ell_delta .+ 1] ./ A_planck^2
    cl_pp_scaled = cl_pp[lik.ell_delta .+ 1]
    
    res = lik.win_delta[:, :, 1] * cl_tt_scaled .+
          lik.win_delta[:, :, 2] * cl_ee_scaled .+
          lik.win_delta[:, :, 3] * cl_te_scaled .+
          lik.win_delta[:, :, 4] * cl_pp_scaled
    return C_b .+ res .- lik.delta_cl_fid
end

function compute_total_binned(lik::MargLensingLikelihood, cl_pp; A_planck=1.0)
    C_b = lik.win_PP * cl_pp[lik.ell_PP .+ 1]
    cl_pp_range = cl_pp[lik.ell_delta .+ 1]
    res = lik.win_delta[:, :, 1] * cl_pp_range
    return C_b .+ res .- lik.delta_cl_fid
end

function loglike(lik::LensingLikelihood, cl_pp, cl_tt, cl_ee, cl_te; A_planck=1.0)
    C_b_total = compute_total_binned(lik, cl_pp, cl_tt, cl_ee, cl_te; A_planck=A_planck)
    Δ = lik.cl_hat .- C_b_total
    return -0.5 * dot(Δ, lik.cov_inv * Δ)
end

function loglike(lik::MargLensingLikelihood, cl_pp; A_planck=1.0)
    C_b_total = compute_total_binned(lik, cl_pp; A_planck=A_planck)
    Δ = lik.cl_hat .- C_b_total
    return -0.5 * dot(Δ, lik.cov_inv * Δ)
end

export LensingLikelihood, MargLensingLikelihood
export PlanckPR4Lensing, PlanckPR4LensingMarged
export loglike, compute_total_binned

end
