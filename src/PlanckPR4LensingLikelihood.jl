module PlanckPR4LensingLikelihood

using LinearAlgebra, DelimitedFiles, Printf, Artifacts, NPZ

# 1. Struct definitions
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

# 2. Internal builders (all artifact resolution happens here)
function _build_lensing()
    data_dir = artifact"planck_pr4_lensing_data"
    function _path(filename)
        p = joinpath(data_dir, "planck_pr4_data_complete", filename)
        isfile(p) || error("Data file not found: $p")
        return p
    end
    lmin, lmax, nbins = 2, 2500, 9
    
    cl_hat = vec(npzread(_path("bandpowers.npy")))
    cov = npzread(_path("covmat.npy"))
    cov_inv = inv(cov)
    
    win_PP = npzread(_path("windows_PP.npy"))
    
    delta_cl_fid = vec(npzread(_path("fid_correction.npy")))
    
    win_delta = permutedims(npzread(_path("win_delta.npy")), (2, 3, 1))
    
    return LensingLikelihood(cl_hat, cov_inv, win_PP, lmin:lmax, delta_cl_fid, win_delta, lmin:lmax, lmin, lmax, nbins)
end

function _build_marged_lensing()
    data_dir = artifact"planck_pr4_lensing_data"
    function _path(filename)
        p = joinpath(data_dir, "planck_pr4_data_complete", filename)
        isfile(p) || error("Data file not found: $p")
        return p
    end
    lmin, lmax, nbins = 2, 2500, 9
    
    cl_hat = vec(npzread(_path("bandpowers.npy")))
    cov = npzread(_path("covmat_marged.npy"))
    cov_inv = inv(cov)
    
    win_PP = npzread(_path("windows_PP.npy"))

    delta_cl_fid = vec(npzread(_path("fid_correction.npy")))

    win_delta = permutedims(npzread(_path("win_delta.npy")), (2, 3, 1))
    
    return MargLensingLikelihood(cl_hat, cov_inv, win_PP, lmin:lmax, delta_cl_fid, win_delta, lmin:lmax, lmin, lmax, nbins)
end

# 3. Module-level Refs (empty at precompile time)
const _lensing_likelihood        = Ref{LensingLikelihood}()
const _marged_lensing_likelihood = Ref{MargLensingLikelihood}()

# 4. __init__: runs once at `using` time, populates Refs
function __init__()
    _lensing_likelihood[]        = _build_lensing()
    _marged_lensing_likelihood[] = _build_marged_lensing()
end

# 5. Public zero-cost accessors
PlanckPR4Lensing()       = _lensing_likelihood[]
PlanckPR4LensingMarged() = _marged_lensing_likelihood[]

# 6. compute_total_binned, loglike — unchanged from current code
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