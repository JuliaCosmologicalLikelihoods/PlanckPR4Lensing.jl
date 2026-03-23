module PlanckPR4LensingLikelihood

using LinearAlgebra, DelimitedFiles, Printf

# ------------------------------------------------------------------ #
#  Data structures                                                     #
# ------------------------------------------------------------------ #

struct LensingLikelihood
    cl_hat::Vector{Float64}         # observed PP bandpowers, length nbins=9
    cov_inv::Matrix{Float64}        # 9×9 inverse covariance
    win_PP::Matrix{Float64}         # (nbins, nell_PP): binning windows for PP
    ell_PP::UnitRange{Int}          # ell range for win_PP columns
    delta_cl_fid::Vector{Float64}   # fiducial N1 correction, length nbins
    win_delta::Array{Float64,3}     # (nbins, nell_delta, 4): N1 windows TT,EE,TE,PP
    ell_delta::UnitRange{Int}       # ell range for win_delta columns
    lmin::Int                       # = 2
    lmax::Int                       # = 2500
    nbins::Int
end

struct MargLensingLikelihood
    cl_hat::Vector{Float64}
    cov_inv::Matrix{Float64}
    win_PP::Matrix{Float64}
    ell_PP::UnitRange{Int}
    delta_cl_fid::Vector{Float64}   # fiducial N1 correction, length nbins
    win_delta::Array{Float64,3}     # (nbins, nell_delta, 1): N1 windows PP
    ell_delta::UnitRange{Int}
    lmin::Int
    lmax::Int
    nbins::Int
end

# ------------------------------------------------------------------ #
#  Covariance matrix reader                                            #
# ------------------------------------------------------------------ #
function read_covmat(path, nbins)
    data = readdlm(path)
    if size(data, 1) == nbins && size(data, 2) == nbins
        return convert(Matrix{Float64}, data)
    end
    return convert(Matrix{Float64}, data)
end

# ------------------------------------------------------------------ #
#  Window file reader                                                  #
# ------------------------------------------------------------------ #
function read_window_files(template::String, nbins::Int, lmin::Int, lmax::Int, ncols::Int)
    nell = lmax - lmin + 1
    W = zeros(nbins, nell, ncols)
    for b in 1:nbins
        path = replace(template, "%u" => string(b))
        data = readdlm(path)
        for i in 1:size(data, 1)
            ell = Int(data[i, 1])
            if lmin <= ell <= lmax
                W[b, ell - lmin + 1, :] = convert(Vector{Float64}, data[i, 2:ncols+1])
            end
        end
    end
    return W
end

# ------------------------------------------------------------------ #
#  Constructors                                                        #
# ------------------------------------------------------------------ #

function PlanckPR4Lensing(data_dir::String)
    lmin, lmax, nbins = 2, 2500, 9
    
    cl_hat_file = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_bandpowers.dat")
    cl_hat = convert(Vector{Float64}, readdlm(cl_hat_file, skipstart=1)[:, 5])
    
    cov_file = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_cov.dat")
    cov_inv = inv(read_covmat(cov_file, nbins))
    
    win_template = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_window/window%u.dat")
    win_PP = read_window_files(win_template, nbins, lmin, lmax, 1)[:, :, 1]
    
    delta_fid_file = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_lensing_fiducial_correction.dat")
    delta_cl_fid = convert(Vector{Float64}, readdlm(delta_fid_file, skipstart=1)[:, 2])
    
    win_delta_template = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_lens_delta_window/window%u.dat")
    win_delta = read_window_files(win_delta_template, nbins, lmin, lmax, 4)
    
    return LensingLikelihood(cl_hat, cov_inv, win_PP, lmin:lmax, delta_cl_fid, win_delta, lmin:lmax, lmin, lmax, nbins)
end

function PlanckPR4LensingMarged(data_dir::String)
    lmin, lmax, nbins = 2, 2500, 9
    
    cl_hat_file = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_bandpowers.dat")
    cl_hat = convert(Vector{Float64}, readdlm(cl_hat_file, skipstart=1)[:, 5])
    
    cov_file = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_CMBmarged_cov.dat")
    cov_inv = inv(read_covmat(cov_file, nbins))
    
    win_template = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_window/window%u.dat")
    win_PP = read_window_files(win_template, nbins, lmin, lmax, 1)[:, :, 1]

    delta_fid_file = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_CMBmarged_lensing_fiducial_correction.dat")
    delta_cl_fid = convert(Vector{Float64}, readdlm(delta_fid_file, skipstart=1)[:, 2])

    win_delta_template = joinpath(data_dir, "pp_consext8_npipe_smicaed_TiPi_jTP_pre30T_kfilt_rdn0cov_PS1_CMBmarged_lens_delta_window/window%u.dat")
    win_delta = read_window_files(win_delta_template, nbins, lmin, lmax, 1)
    
    return MargLensingLikelihood(cl_hat, cov_inv, win_PP, lmin:lmax, delta_cl_fid, win_delta, lmin:lmax, lmin, lmax, nbins)
end

# ------------------------------------------------------------------ #
#  Likelihood evaluation                                               #
# ------------------------------------------------------------------ #

function compute_total_binned(lik::LensingLikelihood,
                              cl_pp::AbstractVector,
                              cl_tt::AbstractVector,
                              cl_ee::AbstractVector,
                              cl_te::AbstractVector;
                              A_planck::Real = 1.0)
    # Standard binning: PP is NOT divided by A_planck^2
    C_b = lik.win_PP * cl_pp[lik.ell_PP .+ 1]
    
    # Linear correction: TT, EE, TE ARE divided by A_planck^2
    # PP in linear correction is NOT divided.
    cl_tt_scaled = cl_tt[lik.ell_delta .+ 1] ./ A_planck^2
    cl_ee_scaled = cl_ee[lik.ell_delta .+ 1] ./ A_planck^2
    cl_te_scaled = cl_te[lik.ell_delta .+ 1] ./ A_planck^2
    cl_pp_scaled = cl_pp[lik.ell_delta .+ 1]
    
    res = zeros(eltype(cl_pp), lik.nbins)
    for b in 1:lik.nbins
        res[b] += dot(lik.win_delta[b, :, 1], cl_tt_scaled)
        res[b] += dot(lik.win_delta[b, :, 2], cl_ee_scaled)
        res[b] += dot(lik.win_delta[b, :, 3], cl_te_scaled)
        res[b] += dot(lik.win_delta[b, :, 4], cl_pp_scaled)
    end
    
    return C_b .+ res .- lik.delta_cl_fid
end

function compute_total_binned(lik::MargLensingLikelihood, cl_pp::AbstractVector; A_planck::Real = 1.0)
    # Marged only uses PP, which is NOT divided by A_planck^2
    C_b = lik.win_PP * cl_pp[lik.ell_PP .+ 1]
    cl_pp_range = cl_pp[lik.ell_delta .+ 1]
    
    res = zeros(eltype(cl_pp), lik.nbins)
    for b in 1:lik.nbins
        res[b] += dot(lik.win_delta[b, :, 1], cl_pp_range)
    end
    
    return C_b .+ res .- lik.delta_cl_fid
end

function loglike(lik::LensingLikelihood,
                 cl_pp::AbstractVector,
                 cl_tt::AbstractVector,
                 cl_ee::AbstractVector,
                 cl_te::AbstractVector;
                 A_planck::Real = 1.0)
    C_b_total = compute_total_binned(lik, cl_pp, cl_tt, cl_ee, cl_te; A_planck=A_planck)
    Δ = lik.cl_hat .- C_b_total
    return -0.5 * dot(Δ, lik.cov_inv * Δ)
end

function loglike(lik::MargLensingLikelihood, cl_pp::AbstractVector; A_planck::Real = 1.0)
    C_b_total = compute_total_binned(lik, cl_pp; A_planck=A_planck)
    Δ = lik.cl_hat .- C_b_total
    return -0.5 * dot(Δ, lik.cov_inv * Δ)
end

export LensingLikelihood, MargLensingLikelihood
export PlanckPR4Lensing, PlanckPR4LensingMarged
export loglike, compute_total_binned

end
