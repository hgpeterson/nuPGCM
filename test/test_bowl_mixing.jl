@testset "Bowl mixing" begin
    h  = 0.1
    α  = 0.5
    bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
    if !isfile(bowl_file)
        include(joinpath(@__DIR__, "../meshes/mesh_bowl3D.jl"))
        generate_bowl_mesh_3D(h, α)
    end

    ε  = 2e-1
    μϱ = 1e1
    N² = 1/α
    f(x) = 1.0 + 0.5*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(; ε, α, μϱ, N², f, H)

    κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
    forcings = Forcings(1.0, κ, κ, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))

    mesh = Mesh(bowl_file)
    fe_data = FEData(mesh;
        u_diri_tags  = ["bottom", "surface"],
        u_diri_masks = [(true,true,true), (false,false,true)],
        b_diri_tags  = ["surface"],
        b_diri_vals  = [x -> 0.0])

    inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)

    @testset "mean pressure constraint (bowl)" begin
        # Bowl mesh is non-periodic: the mean pressure AffineConstraint is active.
        # After invert!, ∫p dΩ must be zero to within solver tolerance.
        b_vec = randn(fe_data.nb)
        invert!(inv_tk, b_vec)
        x_sol = on_architecture(CPU(), inv_tk.solver.x)

        _, cv_p, _ = make_cell_values(fe_data)
        p_range    = dof_range(fe_data.dh_up, :p)
        x_up       = zeros(ndofs(fe_data.dh_up))
        x_up[fe_data.p_dof_indices] .= x_sol[fe_data.p_dof_indices]

        p_integral = 0.0
        vol        = 0.0
        for cc in CellIterator(fe_data.dh_up)
            reinit!(cv_p, cc)
            local_p = x_up[celldofs(cc)[p_range]]
            for q in 1:getnquadpoints(cv_p)
                dΩ = getdetJdV(cv_p, q)
                p_integral += function_value(cv_p, q, local_p) * dΩ
                vol        += dΩ
            end
        end
        # tolerance reflects the small discrepancy between the constraint quadrature
        # (order 3, used in _mean_pressure_constraint) and the test quadrature (order 4)
        @test abs(p_integral) / vol < 1e-3
    end

    Δt = 1e-4 * μϱ / (α*ε)^2
    ts = BDF2(; t_start=0.0, t_stop=50*Δt, Δt)
    evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
    model = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)
    set_out_dir!("/tmp/nuPGCM_regression")
    run!(model; n_info=typemax(Int), n_save=Inf, n_plot=Inf)

    datafile = joinpath(@__DIR__, "data/bowl_mixing.jld2")
    if !isfile(datafile)
        @warn "Reference not found, saving bowl mixing state..."
        save_state(model, datafile)
        @test true
    else
        jldopen(datafile, "r") do d
            @test norm(model.state.u - d["u"]) / norm(d["u"]) < 1e-3
            @test norm(model.state.b - d["b"]) / max(norm(d["b"]), 1e-10) < 1e-3
        end
    end
end
