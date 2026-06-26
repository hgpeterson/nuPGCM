@testset "Inversion" begin
    fe_data  = FE_DATA
    params   = PARAMS
    forcings = FORCINGS
    nu, np, nb = get_n_dofs(fe_data)
    N_up = ndofs(fe_data.dh_up)

    rel_asymm(K) = norm(K - K') / norm(K)

    @testset "build_A_inversion" begin
        A = build_A_inversion(fe_data, params, forcings.ν)
        @test size(A) == (N_up, N_up)
        @test nnz(A) > 0

        # Stokes + Coriolis is non-symmetric (Coriolis is antisymmetric)
        @test rel_asymm(A) > 1e-10

        # u-DOF diagonal entries are non-negative (viscous term dominates)
        @test all(diag(A)[fe_data.u_dof_indices] .>= 0)

        # Coriolis matters: A changes when f = 0
        params_nof = Parameters(; params.ε, params.α, params.μϱ, params.N²,
                                  f=x->0.0, H=params.H)
        @test !(A ≈ build_A_inversion(fe_data, params_nof, forcings.ν))
    end

    @testset "build_B_inversion" begin
        B = build_B_inversion(fe_data, params)
        @test size(B) == (N_up, nb)
        @test nnz(B) > 0

        # pressure rows are zero (only velocity couples to buoyancy)
        @test nnz(B[fe_data.p_dof_indices, :]) == 0

        # linear in 1/α
        params2 = Parameters(; params.ε, α=2*params.α, params.μϱ, params.N²,
                               f=params.f, H=params.H)
        @test build_B_inversion(fe_data, params2) ≈ (params.α/(2*params.α)) * B
    end

    @testset "build_f_wind" begin
        forcings_wind = Forcings(forcings.ν, forcings.κₕ, forcings.κᵥ,
                                  x->0.1, x->0.0, SurfaceDirichletBC(x->0.0))
        f = build_f_wind(fe_data, params, forcings_wind)
        @test length(f) == N_up
        @test norm(f) > 0
        @test norm(f[fe_data.p_dof_indices]) == 0   # only u-rows nonzero

        # zero wind → zero vector
        @test iszero(build_f_wind(fe_data, params, forcings))
    end

    @testset "InversionToolkit" begin
        inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)

        @test size(inv_tk.B)     == (N_up, nb)
        @test length(inv_tk.f_wind) == N_up
        @test length(inv_tk.f_bc)   == N_up
        @test norm(inv_tk.f_bc) == 0   # homogeneous velocity BCs
        @test size(inv_tk.solver.A) == (N_up, N_up)
        @test inv_tk.solver.P isa Factorization

        # invert! produces a nonzero solution
        b_vec = randn(nb)
        invert!(inv_tk, b_vec)
        x = on_architecture(CPU(), inv_tk.solver.x)
        @test norm(x) > 0

        # LU residual is small
        A_cpu = on_architecture(CPU(), inv_tk.solver.A)
        y_cpu = on_architecture(CPU(), inv_tk.solver.y)
        @test norm(A_cpu * x - y_cpu) / norm(y_cpu) < 1e-8
    end

end
