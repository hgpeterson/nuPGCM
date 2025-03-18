using Test
using nuPGCM
using Gridap
using JLD2
using CUDA
using SparseArrays
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

set_out_dir!("./test")

function coarse_inversion(dim, arch)
    # params/funcs
    ε = 1e-1
    α = 1/2
    N² = 1
    params = Parameters(ε, α, 0., N², 0.)
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = 1 - x[1]^2 - x[2]^2
    ν(x) = 1

    # coarse mesh
    h = 0.1
    mesh = Mesh(@sprintf("meshes/bowl%sD_%0.2f.msh", dim, h))
    @assert mesh.dim == dim

    # build inversion matrices and test LHS against saved matrix
    A_inversion_fname = @sprintf("test/data/A_inversion_%sD_%e_%e_%e_%e_%e.h5", dim, h, ε, α, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; ofile=A_inversion_fname)
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
    model = Model(arch, params, mesh, inversion_toolkit)

    # simple test buoyancy field: b = δ exp(-(z + H)/δ)
    set_b!(model, x -> 0.1*exp(-(x[3] + H(x))/0.1))

    # invert
    invert!(model)

    # plot for sanity check
    sim_plots(model, H, 0)

    # compare state with data
    datafile = @sprintf("test/data/inversion_%sD.jld2", dim)
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
    # @testset "2D GPU" begin
    #     coarse_inversion(TwoD(), GPU())
    # end
    # @testset "3D CPU" begin
    #     coarse_inversion(3, CPU())
    # end
    # @testset "3D GPU" begin
    #     coarse_inversion(ThreeD(), GPU())
    # end
end