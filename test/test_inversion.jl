@testset "Inversion" begin
    fe_data  = FE_DATA
    params   = PARAMS
    forcings = FORCINGS
    nu, np, nb = get_n_dofs(fe_data)
    N_up   = ndofs(fe_data.dh_up)
    N_free = length(fe_data.free_dofs)

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

        # B and f_wind are projected onto the free DOFs
        @test size(inv_tk.B) == (N_free, nb)
        @test length(inv_tk.f_wind) == N_free
        @test size(inv_tk.solver.A) == (N_free, N_free)
        @test N_free == N_up - length(fe_data.ch_up.prescribed_dofs)
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

    @testset "BlockTriangularPreconditioner" begin
        # preconditioned FGMRES reproduces the direct solve
        inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
        invert!(inv_tk, randn(nb))
        A, y, x_direct = inv_tk.solver.A, inv_tk.solver.y, inv_tk.solver.x

        P = BlockTriangularPreconditioner(CPU(), fe_data, A,
                CahouetChabardInverse(CPU(), fe_data, params, forcings, zeros(nb)))
        x, stats = Krylov.fgmres(A, y; N=P, memory=100, atol=0.0, rtol=1e-10, itmax=500)
        @test stats.solved
        @test norm(x - x_direct) / norm(x_direct) < 1e-6
    end

    @testset "eddy viscosity cache" begin
        # build_A_visc! against full assembly and CᵀAC, for quadratic b, whose ∂z b
        # is known at every quadrature point
        eddy_param = EddyParameterization(f=x->1.0, N²min=1e-3, ν_min=0.1)
        C   = fe_data.C_up
        A   = C' * build_A_inversion(fe_data, params, 0.0) * C
        lhs = InversionLHSCache(fe_data, A)

        for c in (0.0, 1.0, -10.0)
            b_vec = fill_b(fe_data, x -> c * (x[3]^2 + x[1]*x[2]))
            ν = x -> ν_eddy(eddy_param, eddy_param.f(x), params.α * (params.N² + 2c*x[3]))
            A_ref = C' * build_A_inversion(fe_data, params, ν) * C

            build_A_visc!(A, lhs, params, eddy_param, b_vec)
            @test norm(A - A_ref) / norm(A_ref) < 1e-9
        end
    end

    @testset "InversionToolkit with eddy parameterization" begin
        eddy_param = EddyParameterization(f=x->1.0, N²min=1e-3, ν_min=0.1)
        forcings_eddy = Forcings(forcings.ν, forcings.κₕ, forcings.κᵥ,
                                 forcings.τˣ, forcings.τʸ, SurfaceDirichletBC(x->0.0);
                                 eddy_param)
        inv_tk = InversionToolkit(CPU(), fe_data, params, forcings_eddy)
        @test inv_tk.lhs_cache isa InversionLHSCache

        # a plain (non-eddy) toolkit has no lhs_cache
        inv_tk_plain = InversionToolkit(CPU(), fe_data, params, forcings)
        @test inv_tk_plain.lhs_cache === nothing

        # invert! still works through the eddy-cache-built A
        b_vec = randn(nb)
        invert!(inv_tk, b_vec)
        x = on_architecture(CPU(), inv_tk.solver.x)
        @test norm(x) > 0
        A_cpu = on_architecture(CPU(), inv_tk.solver.A)
        y_cpu = on_architecture(CPU(), inv_tk.solver.y)
        @test norm(A_cpu * x - y_cpu) / norm(y_cpu) < 1e-6
    end

end
