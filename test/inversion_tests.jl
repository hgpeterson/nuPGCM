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
    ε² = 1e-2
    γ = 1/4
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = 1 - x[1]^2 - x[2]^2
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

    # coarse mesh
    h = 0.1
    mesh = Mesh(@sprintf("meshes/bowl%s_%0.2f.msh", dim, h))

    # assemble LHS inversion and test against saved matrix
    A_inversion_fname = @sprintf("test/data/A_inversion_%s_%e_%e_%e_%e_%e.h5", dim, h, ε², γ, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion = nuPGCM.build_A_inversion(mesh, γ, ε², ν, f; fname=A_inversion_fname)
    else
        A_inversion = nuPGCM.build_A_inversion(mesh, γ, ε², ν, f; fname="A_inversion_temp.jld2")
        jldopen(A_inversion_fname, "r") do file
            @test A_inversion ≈ file["A_inversion"]
        end
    end

    # re-order dofs
    A_inversion = A_inversion[mesh.dofs.p_inversion, mesh.dofs.p_inversion]

    # build RHS matrix for inversion
    B_inversion = nuPGCM.build_B_inversion(mesh)

    # re-order dofs
    B_inversion = B_inversion[mesh.dofs.p_inversion, :]

    # preconditioner
    if typeof(arch) == CPU
        P_inversion = lu(A_inversion)
    else
        P_inversion = Diagonal(on_architecture(arch, 1/h^dim.n*ones(size(A_inversion, 1))))
    end

    # move to arch
    A_inversion = on_architecture(arch, A_inversion)
    B_inversion = on_architecture(arch, B_inversion)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion)

    # background state ∂z(b) = N^2
    N² = 1.

    # simple test buoyancy field: b = δ exp(-(z + H)/δ)
    u = interpolate_everywhere(0, mesh.spaces.X_trial[1])
    v = interpolate_everywhere(0, mesh.spaces.X_trial[2])
    w = interpolate_everywhere(0, mesh.spaces.X_trial[3])
    p = interpolate_everywhere(0, mesh.spaces.X_trial[4]) 
    b = interpolate_everywhere(x -> 0.1*exp(-(x[3] + H(x))/0.1), mesh.spaces.B_trial)
    t = 0.
    state = State(u, v, w, p, b, t) 

    # invert
    invert!(inversion_toolkit, b)
    set_state!(state, mesh, inversion_toolkit)

    # plot for sanity check
    sim_plots(dim, u, v, w, b, N², H, 0, 0)

    # compare state with data
    datafile = @sprintf("test/data/inversion_%s.h5", dim)
    if !isfile(datafile)
        @warn "Data file not found, saving state..."
        save(state; ofile=datafile)
    else
        # state_data = load_state(datafile)
        # @test isapprox(state.u, state_data.u, rtol=1e-2)
        # @test isapprox(state.v, state_data.v, rtol=1e-2)
        # @test isapprox(state.w, state_data.w, rtol=1e-2)
        # @test isapprox(state.p, state_data.p, rtol=1e-2)
        # @test isapprox(state.b, state_data.b, rtol=1e-2)
        file = h5open(datafile, "r")
        @test isapprox(state.u, read(file, "ux"), rtol=1e-2)
        @test isapprox(state.v, read(file, "uy"), rtol=1e-2)
        @test isapprox(state.w, read(file, "uz"), rtol=1e-2)
        @test isapprox(state.p, read(file, "p"), rtol=1e-2) #FAILURE HERE
        @test isapprox(state.b, read(file, "b"), rtol=1e-2)
        close(file)
    end

    # remove temporary files
    rm("A_inversion_temp.jld2", force=true)
end

@testset "Inversion Tests" begin
    @testset "2D CPU" begin
        coarse_inversion(TwoD(), CPU())
    end
    # @testset "2D GPU" begin
    #     coarse_inversion(TwoD(), GPU())
    # end
    @testset "3D CPU" begin
        coarse_inversion(ThreeD(), CPU())
    end
    # @testset "3D GPU" begin
    #     coarse_inversion(ThreeD(), GPU())
    # end
end