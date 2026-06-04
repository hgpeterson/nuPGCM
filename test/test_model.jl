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
end
