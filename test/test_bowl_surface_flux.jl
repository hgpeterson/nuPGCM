@testset "Bowl surface flux" begin
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
    f(x) = 1.0
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(; ε, α, μϱ, N², f, H)

    forcings = Forcings(1.0, x->1e-2, x->1e-2, x->0.0, x->0.0,
                        SurfaceFluxBC(x -> 1e-3*sin(π*x[1])))

    mesh = Mesh(bowl_file)
    fe_data = FEData(mesh;
        u_diri_tags  = ["bottom", "surface"],
        u_diri_masks = [(true,true,true), (false,false,true)])

    inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
    Δt = 1e-1
    ts = BDF2(; t_start=0.0, t_stop=50*Δt, Δt)
    evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
    model = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)
    set_out_dir!("/tmp/nuPGCM_regression")
    run!(model; n_info=typemax(Int), n_save=Inf, n_plot=Inf)

    datafile = joinpath(@__DIR__, "data/bowl_surface_flux.jld2")
    if !isfile(datafile)
        @warn "Reference not found, saving bowl surface flux state..."
        save_state(model, datafile)
        @test true
    else
        jldopen(datafile, "r") do d
            @test norm(model.state.u - d["u"]) / norm(d["u"]) < 1e-3
            @test norm(model.state.b - d["b"]) / norm(d["b"]) < 1e-3
        end
    end
end
