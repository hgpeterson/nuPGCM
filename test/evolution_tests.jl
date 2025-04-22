using Test
using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

set_out_dir!("./test")

function coarse_evolution(dim, arch)
    # params/funcs
    ε = 2e-1
    α = 1/2
    μϱ = 1e1
    N² = 1e0/α
    Δt = 1e-4*μϱ/(α*ε)^2
    params = Parameters(ε, α, μϱ, N², Δt)
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
    T = 5e-2*μϱ/(α*ε)^2

    # coarse mesh
    h = 0.1
    mesh = Mesh(@sprintf("meshes/bowl%sD_%e_%e.msh", dim, h, α))

    # build inversion matrices and test LHS against saved matrix
    A_inversion_fname = @sprintf("test/data/A_inversion_%sD_%e_%e_%e_%e_%e.jld2", dim, h, ε, α, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; A_inversion_ofile=A_inversion_fname)
    else
        # A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν)
        # jldopen(A_inversion_fname, "r") do file
        #     @test A_inversion ≈ file["A_inversion"]
        # end
        file = jldopen(A_inversion_fname, "r")
        A_inversion = file["A_inversion"]
        close(file)
        B_inversion = nuPGCM.build_B_inversion(mesh, params)
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

    # build evolution matrices and test against saved matrices
    θ = Δt/2 * (α*ε)^2/μϱ
    A_diff_fname = @sprintf("test/data/A_diff_%sD_%e_%e_%e.jld2", dim, h, θ, α)
    A_adv_fname = @sprintf("test/data/A_adv_%sD_%e.jld2", dim, h)
    if !isfile(A_diff_fname) || !isfile(A_adv_fname)
        @warn "A_diff or A_adv file not found, generating..."
        A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ; 
                                            A_adv_ofile=A_adv_fname, A_diff_ofile=A_diff_fname)
    else
        A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ) 
        jldopen(A_adv_fname, "r") do file
            @test A_adv ≈ file["A_adv"]
        end
        jldopen(A_diff_fname, "r") do file
            @test A_diff ≈ file["A_diff"]
        end
    end

    # re-order dofs
    A_adv  = A_adv[mesh.dofs.p_b, mesh.dofs.p_b]
    A_diff = A_diff[mesh.dofs.p_b, mesh.dofs.p_b]
    B_diff = B_diff[mesh.dofs.p_b, :]
    b_diff = b_diff[mesh.dofs.p_b]

    # preconditioners
    if typeof(arch) == CPU
        P_diff = lu(A_diff)
        P_adv  = lu(A_adv)
    else
        P_diff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_diff))))
        P_adv  = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_adv))))
    end

    # move to arch
    A_adv  = on_architecture(arch, A_adv)
    A_diff = on_architecture(arch, A_diff)
    B_diff = on_architecture(arch, B_diff)
    b_diff = on_architecture(arch, b_diff)

    # setup evolution toolkit
    evolution_toolkit = EvolutionToolkit(A_adv, P_adv, A_diff, P_diff, B_diff, b_diff)

    # put it all together in the `model` struct
    model = rest_state_model(arch, params, mesh, inversion_toolkit, evolution_toolkit)

    # solve
    run!(model, T)

    # plot for sanity check
    sim_plots(model, H, 0)

    # compare state with data
    datafile = @sprintf("test/data/evolution_%sD.jld2", dim)
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

@testset "Evolution Tests" begin
    @testset "2D CPU" begin
        coarse_evolution(2, CPU())
    end
    @testset "2D GPU" begin
        coarse_evolution(2, GPU())
    end
    @testset "3D CPU" begin
        coarse_evolution(3, CPU())
    end
    @testset "3D GPU" begin
        coarse_evolution(3, GPU())
    end
end