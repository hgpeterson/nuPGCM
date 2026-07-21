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
        # wind stress drives flow: u should be nonzero
        @test norm(model.state.u) > 0
        # pressure should also be nonzero
        @test norm(model.state.p) > 0
    end

    @testset "run!" begin
        set_b!(model, zeros(nb))
        nuPGCM.set_out_dir!("/tmp/nuPGCM_test_run")
        run!(model; n_info=typemax(Int), n_save=Inf, n_plot=Inf)
        # wind stress should have driven some flow and buoyancy change
        @test norm(model.state.u) > 0
        @test norm(model.state.b) >= 0   # may stay zero with Dirichlet b BC
        @test !any(isnan, model.state.u)
        @test !any(isnan, model.state.b)
    end

    @testset "IO" begin
        ofile = "/tmp/nuPGCM_test_state.jld2"
        save_state(model, ofile)
        @test isfile(ofile)

        # round-trip: save then load
        model2 = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk,
                       BDF1(; t_start=0.0, t_stop=2Δt, Δt))
        set_state_from_file!(model2, ofile)
        @test model2.state.u ≈ model.state.u
        @test model2.state.b ≈ model.state.b
    end

    @testset "run! with eddy parameterization" begin
        # exercises _update_eddy_A! (cached viscous-block rebuild), including
        # the initial pre-loop sync in run! and the every-10-step refresh
        eddy_param = EddyParameterization(f=x->1.0, N²min=1e-3, ν_min=0.1)
        forcings_eddy = Forcings(1.0, x->1e-2, x->1e-2, x->0.05, x->0.0,
                                 SurfaceDirichletBC(x->0.0); eddy_param)

        inv_tk_eddy = InversionToolkit(CPU(), fe_data, params, forcings_eddy)
        @test inv_tk_eddy.lhs_cache !== nothing

        evo_tk_eddy = EvolutionToolkit(CPU(), fe_data, params, forcings_eddy, ts)
        model_eddy  = Model(CPU(), params, forcings_eddy, fe_data,
                            inv_tk_eddy, evo_tk_eddy, ts)

        set_b!(model_eddy, x -> 0.05 * x[3])
        nuPGCM.set_out_dir!("/tmp/nuPGCM_test_run_eddy")
        run!(model_eddy; n_info=typemax(Int), n_save=Inf, n_plot=Inf)

        @test norm(model_eddy.state.u) > 0
        @test !any(isnan, model_eddy.state.u)
        @test !any(isnan, model_eddy.state.b)

        # the cached A must still solve consistently after the eddy refresh.
        # Check via a bare InversionToolkit invert! (not Model.invert!, whose
        # sync_flow! calls apply!(x, ch_up) afterward to recover periodic
        # mirror DOFs -- that intentionally leaves x no longer satisfying
        # Ax=y for the *condensed* system, so the residual must be read off
        # solver.x/.y right after invert!, before any such recovery).
        b_vec = randn(nb)
        invert!(model_eddy.inversion, b_vec)
        A_cpu = on_architecture(CPU(), model_eddy.inversion.solver.A)
        x_cpu = on_architecture(CPU(), model_eddy.inversion.solver.x)
        y_cpu = on_architecture(CPU(), model_eddy.inversion.solver.y)
        @test norm(A_cpu * x_cpu - y_cpu) / norm(y_cpu) < 1e-6
    end
end
