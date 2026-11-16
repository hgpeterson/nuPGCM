@testset "Model" begin
    fe_data  = FE_DATA
    params   = PARAMS
    forcings = Forcings(1.0, x->1e-2, x->1e-2, x->0.05, x->0.0,
                        SurfaceDirichletBC(x->0.0))
    nu, np, nb = get_n_dofs(fe_data)

    Δt = 1e-3
    ts = BDF1(; t_start=0.0, t_stop=2*Δt, Δt)

    inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
    evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
    model  = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)

    @testset "Construction" begin
        @test model.state isa State
        @test length(model.state.u) == nu
        @test length(model.state.p) == np
        @test length(model.state.b) == nb
        @test all(iszero, model.state.u)
        @test all(iszero, model.state.b)
    end

    @testset "set_b!" begin
        set_b!(model, x -> 0.1 * x[3])
        @test norm(model.state.b) > 0
        set_b!(model, zeros(nb))
        @test all(iszero, model.state.b)
    end

    @testset "invert!" begin
        set_b!(model, x -> 0.01 * x[3])
        invert!(model)
        # wind stress drives flow
        @test norm(model.state.u) > 0
        @test norm(model.state.p) > 0
    end

    @testset "run!" begin
        set_b!(model, zeros(nb))
        set_out_dir!(mktempdir())
        run!(model; n_info=typemax(Int), n_save=Inf)
        @test norm(model.state.u) > 0
        @test all(isfinite, model.state.u)
        @test all(isfinite, model.state.b)
    end

    @testset "update_Δt!" begin
        h_min = minimum(compute_h_cells(fe_data.mesh))

        # at rest the CFL limit is infinite, so the timestep is the maximum one
        u_zero = zeros(nu)
        @test all(iszero, nuPGCM.max_cell_speeds(fe_data, u_zero))
        ts_a = BDF2(; t_start=0.0, t_stop=1.0, Δt=1.0, adaptive=true, CFL_factor=0.5)
        nuPGCM.update_Δt!(ts_a, fe_data, u_zero, h_min)
        @test ts_a.Δt[] == 1.0

        # uniform velocity field
        x_up = nuPGCM.to_up_vec(fe_data, zeros(nu))
        apply_analytical!(x_up, fe_data.dh_up, :u, x -> Vec{3}((3.0, 4.0, 0.0)))
        u_uniform = x_up[fe_data.u_dof_indices]
        speeds = nuPGCM.max_cell_speeds(fe_data, u_uniform)
        @test length(speeds) == getncells(fe_data.mesh.grid)
        @test all(s -> isapprox(s, 5.0; atol=1e-8), speeds)   # |(3,4,0)| = 5
        nuPGCM.update_Δt!(ts_a, fe_data, u_uniform, h_min)
        @test ts_a.Δt[] ≈ 0.5 / fe_data.b_order * h_min / 5.0

        # non-adaptive timestepper leaves Δt untouched
        ts_fixed = BDF1(; t_start=0.0, t_stop=1.0, Δt=0.123, adaptive=false)
        nuPGCM.update_Δt!(ts_fixed, fe_data, u_uniform, h_min)
        @test ts_fixed.Δt[] == 0.123
    end

    @testset "IO" begin
        ofile = joinpath(mktempdir(), "state.jld2")
        save_state(model, ofile)
        model2 = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk,
                       BDF1(; t_start=0.0, t_stop=2Δt, Δt))
        set_state_from_file!(model2, ofile)
        @test model2.state.u ≈ model.state.u
        @test model2.state.b ≈ model.state.b
    end

    @testset "run! with eddy parameterization" begin
        # exercises the cached viscous-block rebuild (update_eddy_A!)
        eddy_param = EddyParameterization(f=x->1.0, N²min=1e-3, ν_min=0.1)
        forcings_eddy = Forcings(1.0, x->1e-2, x->1e-2, x->0.05, x->0.0,
                                 SurfaceDirichletBC(x->0.0); eddy_param)

        inv_tk_eddy = InversionToolkit(CPU(), fe_data, params, forcings_eddy)
        @test inv_tk_eddy.lhs_cache !== nothing

        evo_tk_eddy = EvolutionToolkit(CPU(), fe_data, params, forcings_eddy, ts)
        model_eddy  = Model(CPU(), params, forcings_eddy, fe_data,
                            inv_tk_eddy, evo_tk_eddy, ts)

        set_b!(model_eddy, x -> 0.05 * x[3])
        run!(model_eddy; n_info=typemax(Int), n_save=Inf)

        @test norm(model_eddy.state.u) > 0
        @test all(isfinite, model_eddy.state.u)
        @test all(isfinite, model_eddy.state.b)

        # the cached A must still solve consistently after the eddy refresh
        b_vec = randn(nb)
        invert!(model_eddy.inversion, b_vec)
        A_cpu = on_architecture(CPU(), model_eddy.inversion.solver.A)
        x_cpu = on_architecture(CPU(), model_eddy.inversion.solver.x)
        y_cpu = on_architecture(CPU(), model_eddy.inversion.solver.y)
        @test norm(A_cpu * x_cpu - y_cpu) / norm(y_cpu) < 1e-6
    end
end
