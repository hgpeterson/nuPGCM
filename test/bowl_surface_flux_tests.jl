using Test
using nuPGCM
using JLD2
using Printf

set_out_dir!(@__DIR__)

function bowl_surface_flux(arch)
    # params/funcs
    ε = √1e-1
    α = 1/2
    μϱ = 1
    N² = 0
    Δt = 1e-1
    f₀ = 1
    β = 0
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(ε, α, μϱ, N², Δt, f, H)
    ν = 1
    κₕ(x) = 1e-2 #+ exp(-(x[3] + H(x))/(0.1*α))
    κᵥ(x) = 1e-2 #+ exp(-(x[3] + H(x))/(0.1*α))
    τˣ(x) = 0
    τʸ(x) = 0
    b_surface_flux(x) = 1e-3*sin(π*x[1])
    b_surface_bc = SurfaceFluxBC(b_surface_flux)
    forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
    n_steps = 500

    # coarse mesh
    dim = 3
    h = 0.1
    mesh = Mesh(joinpath(@__DIR__, @sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α)))

    # FE data
    u_diri = Dict("bottom"=>0, "coastline"=>0)
    v_diri = Dict("bottom"=>0, "coastline"=>0)
    w_diri = Dict("bottom"=>0, "coastline"=>0, "surface"=>0)
    b_diri = Dict()
    spaces = Spaces(mesh, u_diri, v_diri, w_diri, b_diri) 
    fe_data = FEData(mesh, spaces)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings)

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    # initial condition
    set_b!(model, x -> x[3]/α)

    # solve
    run!(model; n_steps)

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
            v_data = file["v"]
            w_data = file["w"]
            p_data = file["p"]
            b_data = file["b"]
            @test isapprox(model.state.u.free_values, u_data, rtol=1e-2)
            @test isapprox(model.state.v.free_values, v_data, rtol=1e-2)
            @test isapprox(model.state.w.free_values, w_data, rtol=1e-2)
            @test isapprox(model.state.p.free_values, p_data, rtol=1e-2)
            @test isapprox(model.state.b.free_values, b_data, rtol=1e-2)
        end
    end
end

@testset "Bowl Surface Flux Tests" begin
    @testset "CPU" begin
        bowl_surface_flux(CPU())
    end
    # @testset "GPU" begin
    #     bowl_surface_flux(GPU())
    # end
end