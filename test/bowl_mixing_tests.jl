using Test
using nuPGCM
using JLD2
using Printf

set_out_dir!(@__DIR__)

function bowl_mixing(dim, arch)
    # params/funcs
    ε = 2e-1
    α = 1/2
    μϱ = 1e1
    N² = 1/α
    Δt = 1e-4*μϱ/(α*ε)^2
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    params = Parameters(ε, α, μϱ, N², Δt, f, H)
    ν = 1
    κₕ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
    κᵥ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
    τˣ(x) = 0
    τʸ(x) = 0
    b_surface(x) = 0
    b_surface_bc = SurfaceDirichletBC(b_surface)
    forcings = Forcings(ν, κₕ, κᵥ, τˣ, τʸ, b_surface_bc)
    n_steps = 500

    # coarse mesh
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

    # test inversion matrix
    datafile = @sprintf("%s/data/A_bowl_mixing_%sD.jld2", out_dir, dim)
    if !isfile(datafile)
        @warn "$datafile file not found, generating..."
        iperm = fe_data.dofs.inv_p_inversion
        A_inversion = on_architecture(CPU(), inversion_toolkit.solver.A[iperm, iperm])
        jldsave(datafile; A_inversion, iperm)
    else
        jldopen(datafile, "r") do file
            iperm = fe_data.dofs.inv_p_inversion
            @test iperm == file["iperm"]  # not a big deal if the permutation doesn't match as long as it's consistent
            A_inversion = on_architecture(CPU(), inversion_toolkit.solver.A)
            @test A_inversion[iperm, iperm] ≈ file["A_inversion"]
        end
    end

    # build evolution system
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings) 

    # put it all together in the `model` struct
    model = Model(arch, params, forcings, fe_data, inversion_toolkit, evolution_toolkit)

    # solve
    run!(model; n_steps)

    # plot for sanity check
    save_vtk(model)

    # compare state with data
    datafile = @sprintf("%s/data/bowl_mixing_%sD.jld2", out_dir, dim)
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
            @test isapprox(model.state.u.free_values⋅nuPGCM.x⃗, u_data, rtol=1e-2)
            @test isapprox(model.state.u.free_values⋅nuPGCM.y⃗, v_data, rtol=1e-2)
            @test isapprox(model.state.u.free_values⋅nuPGCM.z⃗, w_data, rtol=1e-2)
            @test isapprox(model.state.p.free_values, p_data, rtol=1e-2)
            @test isapprox(model.state.b.free_values, b_data, rtol=1e-2)
        end
    end
end

@testset "Bowl Mixing Tests" begin
    @testset "2D CPU" begin
        bowl_mixing(2, CPU())
    end
    @testset "2D GPU" begin
        bowl_mixing(2, GPU())
    end
    @testset "3D CPU" begin
        bowl_mixing(3, CPU())
    end
    @testset "3D GPU" begin
        bowl_mixing(3, GPU())
    end
end