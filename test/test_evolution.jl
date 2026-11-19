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

    @testset "convection" begin
        # update_evolution_LHS! against full assembly and CᵀAC, for quadratic b, whose
        # ∂z b is known at every quadrature point, and nonzero Dirichlet values g
        fe_g = FEData(fe_data.mesh;
            u_diri_tags  = ["bottom", "surface"],
            u_diri_masks = [(true,true,true), (false,false,true)],
            b_diri_tags  = ["surface"],
            b_diri_vals  = [x -> x[1] - 0.5*x[2]])
        conv_param = ConvectionParameterization(κᶜ=1.0, N²min=1e-1)
        forcings_conv = Forcings(1.0, κₕ, κᵥ, x -> 0.0, x -> 0.0, SurfaceDirichletBC(x -> 0.0);
                                 conv_param)
        ts  = BDF1(; t_start=0.0, t_stop=1e-2, Δt=1e-3)
        evo = EvolutionToolkit(CPU(), fe_g, params, forcings_conv, ts)
        θ   = nuPGCM.evolution_parameter(params, ts)
        C, g = evo.C, evo.g
        @test norm(g) > 0
        M  = build_M(fe_g)
        Kₕ = build_Kₕ(fe_g, κₕ)

        for c in (0.0, 1.0, -10.0)
            b_vec = fill_b(fe_g, x -> c * x[3]^2)
            κ(x) = κᵥ(x) + nuPGCM.κ_conv_extra(conv_param, params.α * (params.N² + 2c*x[3]))
            nuPGCM.update_evolution_LHS!(evo, params, forcings_conv, ts, b_vec)

            Kᵥ    = build_Kᵥ(fe_g, κ)
            A_ref = C' * (M + θ * (Kₕ + Kᵥ)) * C
            @test norm(evo.solver.A - A_ref) / norm(A_ref) < 1e-9
            rhs_ref = build_rhs_diff(params, fe_g, κ) - (Kₕ + Kᵥ) * g
            @test norm(evo.rhs_diff - rhs_ref) / norm(rhs_ref) < 1e-9
        end
    end
end
