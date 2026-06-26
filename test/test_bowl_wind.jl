@testset "Bowl wind stress" begin
    h  = 0.1
    α  = 0.5
    bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
    if !isfile(bowl_file)
        include(joinpath(@__DIR__, "../meshes/mesh_bowl3D.jl"))
        generate_bowl_mesh_3D(h, α)
    end

    ε  = sqrt(1e-1)
    μϱ = 1.0
    N² = 0.0
    f(x) = 0.5*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(; ε, α, μϱ, N², f, H)

    κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
    forcings = Forcings(1.0, κ, κ,
                        x -> -1e-1*cos(π*x[2]/2), x->0.0,
                        SurfaceDirichletBC(x->0.0))

    mesh = Mesh(bowl_file)
    fe_data = FEData(mesh;
        u_diri_tags  = ["bottom", "surface"],
        u_diri_masks = [(true,true,true), (false,false,true)],
        b_diri_tags  = ["surface"],
        b_diri_vals  = [x -> 0.0])

    inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
    Δt = 1e-4 * μϱ / (α*ε)^2
    ts = BDF1(; t_start=0.0, t_stop=20*Δt, Δt)
    evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
    model = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)
    set_out_dir!("/tmp/nuPGCM_regression")
    run!(model; n_info=typemax(Int), n_save=Inf, n_plot=Inf)

    datafile = joinpath(@__DIR__, "data/bowl_wind.jld2")
    if !isfile(datafile)
        @warn "Reference not found, saving bowl wind state..."
        save_state(model, datafile)
        @test true
    else
        jldopen(datafile, "r") do d
            @test norm(model.state.u - d["u"]) / norm(d["u"]) < 1e-3
            @test norm(model.state.b - d["b"]) / max(norm(d["b"]), 1e-10) < 1e-3
        end
    end
end
