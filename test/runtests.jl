using PlanckPR4LensingLikelihood
using Test, NPZ, LinearAlgebra
using DifferentiationInterface
import ForwardDiff, Zygote, Mooncake, Artifacts

const REF_DIR = joinpath(Artifacts.artifact"planck_pr4_lensing_data", "planck_pr4_data_complete")

@testset "PlanckPR4Lensing.jl" begin

    @testset "Artifact availability" begin
        data_dir = Artifacts.artifact"planck_pr4_lensing_data"
        @test isdir(data_dir)
        @test isfile(joinpath(data_dir, "planck_pr4_data_complete", "bandpowers.npy"))
    end

    @testset "No-argument constructors work" begin
        lik  = PlanckPR4Lensing()
        likm = PlanckPR4LensingMarged()
        @test length(lik.cl_hat)  == 9
        @test length(likm.cl_hat) == 9
        @test size(lik.cov_inv)   == (9, 9)
        @test size(likm.cov_inv)  == (9, 9)
    end

    @testset "Log-likelihood (full, A_planck=1)" begin
        lik = PlanckPR4Lensing()
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        cl_tt = npzread(joinpath(REF_DIR, "reference_cl_tt.npy"))
        cl_ee = npzread(joinpath(REF_DIR, "reference_cl_ee.npy"))
        cl_te = npzread(joinpath(REF_DIR, "reference_cl_te.npy"))
        ref_ll = npzread(joinpath(REF_DIR, "reference_loglike_full.npy"))[1]

        ll = loglike(lik, cl_pp, cl_tt, cl_ee, cl_te; A_planck=1.0)
        @test ll ≈ ref_ll rtol=1e-6
    end

    @testset "Log-likelihood (full, A_planck perturbed)" begin
        lik = PlanckPR4Lensing()
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        cl_tt = npzread(joinpath(REF_DIR, "reference_cl_tt.npy"))
        cl_ee = npzread(joinpath(REF_DIR, "reference_cl_ee.npy"))
        cl_te = npzread(joinpath(REF_DIR, "reference_cl_te.npy"))
        ref_ll = npzread(joinpath(REF_DIR, "reference_loglike_full_Ap098.npy"))[1]

        ll = loglike(lik, cl_pp, cl_tt, cl_ee, cl_te; A_planck=0.98)
        @test ll ≈ ref_ll rtol=1e-6
    end

    @testset "Log-likelihood (CMBmarged)" begin
        lik_m = PlanckPR4LensingMarged()
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        
        ll = loglike(lik_m, cl_pp; A_planck=1.0)
        @test isfinite(ll)
        @test ll ≈ -3624.1488382158745 rtol=1e-6
    end

    @testset "Differentiation (Full Likelihood)" begin
        lik = PlanckPR4Lensing()
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        cl_tt = npzread(joinpath(REF_DIR, "reference_cl_tt.npy"))
        cl_ee = npzread(joinpath(REF_DIR, "reference_cl_ee.npy"))
        cl_te = npzread(joinpath(REF_DIR, "reference_cl_te.npy"))
        
        f(x) = loglike(lik, x, cl_tt, cl_ee, cl_te; A_planck=1.0)
        
        backends = [
            AutoForwardDiff(),
            AutoZygote(),
            AutoMooncake(; config=Mooncake.Config())
        ]
        
        for backend in backends
            @testset "Backend: $backend" begin
                g = gradient(f, backend, cl_pp)
                @test all(isfinite, g)
                @test any(x -> x != 0, g)
                
                if backend != AutoForwardDiff()
                    g_fd = gradient(f, AutoForwardDiff(), cl_pp)
                    @test g ≈ g_fd rtol=1e-8
                end
            end
        end
    end

    @testset "Differentiation (Marged Likelihood)" begin
        lik_m = PlanckPR4LensingMarged()
        cl_pp = npzread(joinpath(REF_DIR, "reference_cl_pp.npy"))
        
        f(x) = loglike(lik_m, x; A_planck=1.0)
        
        backends = [
            AutoForwardDiff(),
            AutoZygote(),
            AutoMooncake(; config=Mooncake.Config())
        ]
        
        for backend in backends
            @testset "Backend: $backend" begin
                g = gradient(f, backend, cl_pp)
                @test all(isfinite, g)
                @test any(x -> x != 0, g)
                
                if backend != AutoForwardDiff()
                    g_fd = gradient(f, AutoForwardDiff(), cl_pp)
                    @test g ≈ g_fd rtol=1e-8
                end
            end
        end
    end

end
