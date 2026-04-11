using Test
using nuPGCM
using Gridap
using JLD2
using Printf

set_out_dir!(@__DIR__)

function bowl_surface_flux(arch)
    println()
    @info "Running bowl surface flux test with:" arch
    println()

    # params/funcs
    ε = √1e-1
    α = 1/2
    μϱ = 1
    N² = 0
    f₀ = 1
    β = 0
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(; ε, α, μϱ, N², f, H)
    ν = 1
    κₕ(x) = 1e-2 #+ exp(-(x[3] + H(x))/(0.1*α))
    κᵥ(x) = 1e-2 #+ exp(-(x[3] + H(x))/(0.1*α))
    τˣ(x) = 0
    τʸ(x) = 0
    b_surface_flux(x) = 1e-3*sin(π*x[1])
    b_surface_bc = SurfaceFluxBC(b_surface_flux)
    forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)

    # coarse mesh
    dim = 3
    h = 0.1
    mesh = Mesh(joinpath(@__DIR__, @sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α)))

    # FE data
    u_diri_tags = ["bottom", "coastline", "surface"]
    u_diri_vals = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    u_diri_masks = [(true, true, true), (true, true, true), (false, false, true)]
    spaces = Spaces(mesh; u_diri_tags, u_diri_vals, u_diri_masks) 
    fe_data = FEData(mesh, spaces)

    # timestepper
    Δt = 1e-1
    timestepper = BDF2(; t_start=0, t_stop=50*Δt, Δt)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings, timestepper) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit, timestepper)

    # initial condition
    set_b!(model, x -> x[3]/α)

    # solve
    run!(model)

    # # plot for sanity check
    # save_vtk(model)

    # compare state with data
    datafile = @sprintf("%s/data/bowl_surface_flux.jld2", out_dir)
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
            # p0 = FEFunction(P_trial, p_data)
            b0 = FEFunction(B_trial, b_data)
            u = model.state.u
            # p = model.state.p
            b = model.state.b
            dΩ = model.fe_data.mesh.dΩ
            @test sum(∫( (u - u0)⋅(u - u0) )dΩ)/sum(∫( u0⋅u0 )dΩ) < 1e-3
            # @test sum(∫( (p - p0)*(p - p0) )dΩ)/sum(∫( p0*p0 )dΩ) < 1e-3
            @test sum(∫( (b - b0)*(b - b0) )dΩ)/sum(∫( b0*b0 )dΩ) < 1e-3
        end
    end
end

@testset "Bowl Surface Flux Tests" begin
    @testset "CPU" begin
        bowl_surface_flux(CPU())
    end
end