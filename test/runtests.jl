using PlanckPR4LensingLikelihood
using Test, NPZ, LinearAlgebra

const DATA_DIR = joinpath(@__DIR__, "..", "..", "planck_PR4_lensing", "planckpr4lensing", "data_pr4")
const REF_DIR = joinpath(@__DIR__, "..", "reference_data")

@testset "PlanckPR4Lensing Julia" begin

    @testset "Data loading" begin
        lik = PlanckPR4Lensing(DATA_DIR)

        @test length(lik.cl_hat) == 9
        @test size(lik.cov_inv) == (9, 9)
        @test isapprox(inv(lik.cov_inv), inv(lik.cov_inv)', atol=1e-20)
        @test size(lik.win_PP) == (9, lik.lmax - lik.lmin + 1)
        @test size(lik.win_delta) == (9, lik.lmax - lik.lmin + 1, 4)
        @test length(lik.delta_cl_fid) == 9

        # Values must match Python-extracted reference
        ref_bp = npzread(joinpath(REF_DIR, "bandpowers.npy"))
        # Python bandpowers might be different shape if it includes other things, 
        # but CMBlikes.bandpowers is (ncl, nbins_used). 
        # In our case ncl=1 (PP only).
        @test lik.cl_hat ≈ vec(ref_bp) rtol=1e-10

        ref_cov = npzread(joinpath(REF_DIR, "covmat.npy"))
        @test inv(lik.cov_inv) ≈ ref_cov rtol=1e-8
    end

    @testset "Binning windows" begin
        lik = PlanckPR4Lensing(DATA_DIR)
        ref_windows = npzread(joinpath(REF_DIR, "windows_PP.npy"))  # shape (9, nell)
        @test lik.win_PP ≈ ref_windows rtol=1e-10
    end

    @testset "Theory binning (PP only)" begin
        lik = PlanckPR4Lensing(DATA_DIR)
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        ref_binned = npzread(joinpath(REF_DIR, "reference_binned_pp.npy"))

        cl_pp_range = cl_pp[lik.ell_PP .+ 1] # 0-indexed in python, 1-indexed in julia. 
        # But wait, our dls['pp'] was already padded from 0.
        # So dls['pp'][ells] matches ells.
        # lik.ell_PP is 2:2500.
        # cl_pp is length 2501. cl_pp[1] is L=0, cl_pp[2] is L=1, cl_pp[3] is L=2.
        # So cl_pp[lik.ell_PP .+ 1] is correct.
        
        binned = lik.win_PP * cl_pp[lik.ell_PP .+ 1]
        @test binned ≈ vec(ref_binned) rtol=1e-8
    end

    @testset "Log-likelihood (full, A_planck=1)" begin
        lik = PlanckPR4Lensing(DATA_DIR)
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        cl_tt = npzread(joinpath(REF_DIR, "reference_cl_tt.npy"))
        cl_ee = npzread(joinpath(REF_DIR, "reference_cl_ee.npy"))
        cl_te = npzread(joinpath(REF_DIR, "reference_cl_te.npy"))
        ref_ll = npzread(joinpath(REF_DIR, "reference_loglike_full.npy"))[1]

        ll = loglike(lik, cl_pp, cl_tt, cl_ee, cl_te; A_planck=1.0)
        @test ll ≈ ref_ll rtol=1e-6
    end

    @testset "Log-likelihood (full, A_planck perturbed)" begin
        lik = PlanckPR4Lensing(DATA_DIR)
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        cl_tt = npzread(joinpath(REF_DIR, "reference_cl_tt.npy"))
        cl_ee = npzread(joinpath(REF_DIR, "reference_cl_ee.npy"))
        cl_te = npzread(joinpath(REF_DIR, "reference_cl_te.npy"))
        ref_ll = npzread(joinpath(REF_DIR, "reference_loglike_full_Ap098.npy"))[1]

        ll = loglike(lik, cl_pp, cl_tt, cl_ee, cl_te; A_planck=0.98)
        @test ll ≈ ref_ll rtol=1e-6
    end

    @testset "Log-likelihood (CMBmarged)" begin
        lik_m = PlanckPR4LensingMarged(DATA_DIR)
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        ref_ll = npzread(joinpath(REF_DIR, "reference_loglike_marged.npy"))[1]

        ll = loglike(lik_m, cl_pp; A_planck=1.0)
        @test ll ≈ ref_ll rtol=1e-6
    end

    @testset "Gradient (AD-readiness)" begin
        using ForwardDiff
        lik_m = PlanckPR4LensingMarged(DATA_DIR)
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        
        # We need a wrapper that takes only the parameters we want to differentiate
        f(x) = loglike(lik_m, x; A_planck=1.0)
        
        # ForwardDiff needs a vector of same length as input
        # Our loglike takes the full cl_pp (length 2501)
        # But we only use ell_PP (2:2500)
        
        # Let's make a more surgical test
        g = ForwardDiff.gradient(f, cl_pp)
        @test all(isfinite, g)
        @test any(x -> x != 0, g)
    end

end
