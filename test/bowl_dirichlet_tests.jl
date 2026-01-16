using Test
using nuPGCM
using JLD2
using Printf

set_out_dir!(@__DIR__)

function bowl_dirichlet(arch)
    # params/funcs
    ε = √1e-1
    α = 1/2
    μϱ = 1
    N² = 0
    Δt = 1e-1
    f₀ = 0
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(ε, α, μϱ, N², Δt, f, H)
    ν = 1
    κₕ(x) = 1
    κᵥ(x) = 1
    τˣ(x) = 0
    τʸ(x) = 0
    b_surface(x) = x[2]
    b_surface_bc = SurfaceDirichletBC(b_surface)
    forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
    n_steps = 50

    # coarse mesh
    dim = 3
    h = 0.1
    mesh = Mesh(joinpath(@__DIR__, @sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α)))

    # FE data
    u_diri_tags = ["bottom", "coastline", "surface"]
    u_diri_vals = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    u_diri_masks = [(true, true, true), (true, true, true), (false, false, true)]
    b_diri_tags = ["coastline", "surface"]
    b_diri_vals = [b_surface, b_surface]
    spaces = Spaces(mesh; u_diri_tags, u_diri_vals, u_diri_masks, b_diri_tags, b_diri_vals) 
    fe_data = FEData(mesh, spaces)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    # initial condition
    set_b!(model, b_surface)

    # solve
    run!(model; n_steps)

    # plot for sanity check
    save_vtk(model)

    # compare state with data
    datafile = @sprintf("%s/data/bowl_diri.jld2", out_dir)
    if !isfile(datafile)
        @warn "Data file not found, saving state..."
        save_state(model, datafile)
    else
        jldopen(datafile, "r") do file
            u_data = file["u"]
            p_data = file["p"]
            b_data = file["b"]
            U_trial, P_trial = model.fe_data.spaces.X_trial
            B_trial = model.fe_data.spaces.B_trial
            u0 = FEFunction(U_trial, u_data)
            p0 = FEFunction(P_trial, p_data)
            b0 = FEFunction(B_trial, b_data)
            u = model.state.u
            p = model.state.p
            b = model.state.b
            dΩ = model.fe_data.mesh.dΩ
            @test sum(∫( (u - u0)⋅(u - u0) )dΩ)/sum(∫( u0⋅u0 )dΩ) < 1e-3
            # @test sum(∫( (p - p0)*(p - p0) )dΩ)/sum(∫( p0*p0 )dΩ) < 1e-3
            @test sum(∫( (b - b0)*(b - b0) )dΩ)/sum(∫( b0*b0 )dΩ) < 1e-3
        end
    end
end

@testset "Bowl Dirichlet Tests" begin
    @testset "CPU" begin
        bowl_dirichlet(CPU())
    end
end