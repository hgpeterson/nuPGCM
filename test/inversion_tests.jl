using Test
using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

set_out_dir!("./test")

function coarse_inversion(dim, arch)
    # params/funcs
    ε = 2e-1
    α = 1/2
    N² = 1e0/α
    params = Parameters(ε, α, 0., N², 0.)
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    ν(x) = 1

    # coarse mesh
    h = 0.1
    mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))
    @assert mesh.dim == dim

    # build inversion matrices and test LHS against saved matrix
    A_inversion_fname = @sprintf("data/A_inversion_%sD_%e_%e_%e_%e_%e.jld2", dim, h, ε, α, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; A_inversion_ofile=A_inversion_fname)
    else
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν)
        jldopen(A_inversion_fname, "r") do file
            @test A_inversion ≈ file["A_inversion"]
        end
    end

    # re-order dofs
    A_inversion = A_inversion[mesh.dofs.p_inversion, mesh.dofs.p_inversion]
    B_inversion = B_inversion[mesh.dofs.p_inversion, :]

    # preconditioner
    if typeof(arch) == CPU
        P_inversion = lu(A_inversion)
    else
        P_inversion = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A_inversion, 1))))
    end

    # move to arch
    A_inversion = on_architecture(arch, A_inversion)
    B_inversion = on_architecture(arch, B_inversion)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion)

    # model
    model = inversion_model(arch, params, mesh, inversion_toolkit)

    # simple test buoyancy field: b = δ exp(-(z + H)/(α*δ))
    set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/(0.1*α)))

    # invert
    invert!(model)

    # # plot for sanity check
    # sim_plots(model, H, 0)

    # compare state with data
    datafile = @sprintf("data/inversion_%sD.jld2", dim)
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

@testset "Inversion Tests" begin
    @testset "2D CPU" begin
        coarse_inversion(2, CPU())
    end
    @testset "2D GPU" begin
        coarse_inversion(2, GPU())
    end
    @testset "3D CPU" begin
        coarse_inversion(3, CPU())
    end
    @testset "3D GPU" begin
        coarse_inversion(3, GPU())
    end
end