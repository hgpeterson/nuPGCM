using Test
using nuPGCM
using JLD2
using LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

set_out_dir!(@__DIR__)

function coarse_evolution(dim, arch)
    # params/funcs
    ε = 2e-1
    α = 1/2
    μϱ = 1e1
    N² = 1/α
    Δt = 1e-4*μϱ/(α*ε)^2
    params = Parameters(ε, α, μϱ, N², Δt)
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    H(x) = α*(1 - x[1]^2 - x[2]^2)
    ν = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))
    τˣ(x) = 0
    τʸ(x) = 0
    b₀(x) = 0
    n_steps = 500

    # coarse mesh
    h = 0.1
    mesh = Mesh(joinpath(@__DIR__, @sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α)))

    # FE data
    spaces = Spaces(mesh, b₀)
    fe_data = FEData(mesh, spaces)

    # build inversion matrices and test LHS against saved matrix
    A_inversion_fname = @sprintf("%s/data/A_inversion_%sD_%e_%e_%e_%e_%e.jld2", out_dir, dim, h, ε, α, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion, b_inversion = build_inversion_matrices(fe_data, params, f, ν, τˣ, τʸ; A_inversion_ofile=A_inversion_fname)
    else
        file = jldopen(A_inversion_fname, "r")
        A_inversion = file["A_inversion"]
        close(file)
        B_inversion = nuPGCM.build_B_inversion(fe_data, params)
        b_inversion = nuPGCM.build_b_inversion(fe_data, params, τˣ, τʸ)
    end

    # re-order dofs
    A_inversion = A_inversion[fe_data.dofs.p_inversion, fe_data.dofs.p_inversion]
    B_inversion = B_inversion[fe_data.dofs.p_inversion, :]
    b_inversion = b_inversion[fe_data.dofs.p_inversion]

    # preconditioner
    if typeof(arch) == CPU
        P_inversion = lu(A_inversion)
    else
        P_inversion = Diagonal(on_architecture(arch, 1/h^dim*ones(size(A_inversion, 1))))
    end

    # move to arch
    A_inversion = on_architecture(arch, A_inversion)
    B_inversion = on_architecture(arch, B_inversion)
    b_inversion = on_architecture(arch, b_inversion)

    # setup inversion toolkit
    inversion_toolkit = InversionToolkit(A_inversion, P_inversion, B_inversion, b_inversion)

    # build evolution matrices and test against saved matrices
    θ = Δt/2 * (α*ε)^2/μϱ
    filename = @sprintf("%s/data/evolution_matrices_%sD_%e_%e_%e.jld2", out_dir, dim, h, θ, α)
    if !isfile(filename)
        @warn "Evolution system file not found, generating..."
        A_adv, A_diff, B_diff, b_diff = build_evolution_system(fe_data, params, κ; filename)
    else
        A_adv, A_diff, B_diff, b_diff = build_evolution_system(fe_data, params, κ; force_build=true)
        file = jldopen(filename, "r")
        @test A_adv ≈ file["A_adv"]
        @test A_diff ≈ file["A_diff"]
        @test B_diff ≈ file["B_diff"]
        @test b_diff ≈ file["b_diff"]
        close(file)
    end

    # re-order dofs
    A_adv  =  A_adv[fe_data.dofs.p_b, fe_data.dofs.p_b]
    A_diff = A_diff[fe_data.dofs.p_b, fe_data.dofs.p_b]
    B_diff = B_diff[fe_data.dofs.p_b, :]
    b_diff = b_diff[fe_data.dofs.p_b]

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
    model = rest_state_model(arch, params, fe_data, inversion_toolkit, evolution_toolkit)

    # solve
    run!(model; n_steps)

    # # plot for sanity check
    # sim_plots(model, 0)

    # compare state with data
    datafile = @sprintf("%s/data/evolution_%sD.jld2", out_dir, dim)
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