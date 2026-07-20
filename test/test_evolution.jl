@testset "Evolution" begin
    fe_data  = FE_DATA
    params   = PARAMS
    forcings = FORCINGS
    _, _, nb = get_n_dofs(fe_data)
    κₕ = forcings.κₕ
    κᵥ = forcings.κᵥ

    rel_asymm(K) = norm(K - K') / norm(K)

    @testset "build_M" begin
        M = build_M(fe_data)
        @test size(M) == (nb, nb)
        @test rel_asymm(M) < 1e-14
        @test all(diag(M) .> 0)
        v = randn(nb)
        @test dot(v, M * v) > 0
        @test sum(M * ones(nb)) > 0    # total mass is positive
    end

    @testset "build_Kₕ" begin
        Kₕ = build_Kₕ(fe_data, κₕ)
        @test size(Kₕ) == (nb, nb)
        @test rel_asymm(Kₕ) < 1e-12
        v = randn(nb)
        @test dot(v, Kₕ * v) >= -1e-10 * norm(v)^2

        # linear in κₕ
        @test build_Kₕ(fe_data, x -> 2*κₕ(x)) ≈ 2 * Kₕ
    end

    @testset "build_Kᵥ" begin
        Kᵥ = build_Kᵥ(fe_data, κᵥ)
        @test size(Kᵥ) == (nb, nb)
        @test rel_asymm(Kᵥ) < 1e-12
        v = randn(nb)
        @test dot(v, Kᵥ * v) >= -1e-10 * norm(v)^2

        @test build_Kᵥ(fe_data, x -> 2*κᵥ(x)) ≈ 2 * Kᵥ
        @test !(build_Kᵥ(fe_data, κᵥ) ≈ build_Kₕ(fe_data, κᵥ))
    end

    @testset "build_rhs_diff" begin
        f = build_rhs_diff(params, fe_data, κᵥ)
        @test length(f) == nb
        @test norm(f) > 0

        # linear in N²
        p2 = Parameters(; params.ε, params.α, params.μϱ, N²=2*params.N²,
                          f=params.f, H=params.H)
        @test build_rhs_diff(p2, fe_data, κᵥ) ≈ 2 * f

        # zero when κᵥ = 0
        @test norm(build_rhs_diff(params, fe_data, x->0.0)) == 0
    end

    @testset "build_rhs_flux" begin
        F = 0.5
        f = build_rhs_flux(params, fe_data, SurfaceFluxBC(x->F))
        @test length(f) == nb
        @test norm(f) > 0
        @test build_rhs_flux(params, fe_data, SurfaceFluxBC(x->2F)) ≈ 2 * f
        @test iszero(build_rhs_flux(params, fe_data, SurfaceDirichletBC(x->0.0)))
    end

    @testset "EvolutionToolkit" begin
        ts  = BDF1(; t_start=0.0, t_stop=1e-2, Δt=1e-3)
        evo = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)

        @test size(evo.Kᵥ) == (nb, nb)
        @test size(evo.Kᵥ⁰) == (nb, nb)
        @test length(evo.rhs_diff) == nb
        @test length(evo.rhs_flux) == nb
        @test length(evo.f_bc)     == nb
        @test norm(evo.f_bc) == 0     # homogeneous BCs
        @test size(evo.solver.A) == (nb, nb)
        @test evo.solver.P isa Factorization

        # Kᵥ rebuild changes A
        κᵥ_new(x) = 2 * κᵥ(x)
        evo.Kᵥ.nzval .= build_Kᵥ(fe_data, κᵥ_new).nzval
        collect_evolution_LHS!(evo, params, forcings, ts, fe_data.ch_b)
        A_new = on_architecture(CPU(), evo.solver.A)

        evo2 = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
        @test !(A_new ≈ on_architecture(CPU(), evo2.solver.A))
    end
end
