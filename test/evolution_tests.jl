using Test
using nuPGCM
using Gridap
using Krylov
using CUDA
using CUDA.CUSPARSE
using CUDA.CUSOLVER
using JLD2
using SparseArrays
using LinearAlgebra
using ProgressMeter
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

set_out_dir!("./test")

function coarse_evolution(dim, arch)
    # params/funcs
    ε = 1e-1
    α = 1/2
    μϱ = 1e1
    params = Parameters(ε, α, μϱ)
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    Δt = 1e-4*μϱ/ε^2
    T = 5e-2*μϱ/ε^2
    H(x) = 1 - x[1]^2 - x[2]^2
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

    # coarse mesh
    h = 0.1
    mesh = Mesh(@sprintf("meshes/bowl%s_%0.2f.msh", dim, h))

    # build inversion matrices and test LHS against saved matrix
    A_inversion_fname = @sprintf("test/data/A_inversion_%s_%e_%e_%e_%e_%e.h5", dim, h, ε, α, f₀, β)
    if !isfile(A_inversion_fname)
        @warn "A_inversion file not found, generating..."
        A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν; ofile=A_inversion_fname)
    else
        # A_inversion, B_inversion = build_inversion_matrices(mesh, params, f, ν)
        # jldopen(A_inversion_fname, "r") do file
        #     @test A_inversion ≈ file["A_inversion"]
        # end
        file = jldopen(A_inversion_fname, "r")
        A_inversion = file["A_inversion"]
        close(file)
        B_inversion = nuPGCM.build_B_inversion(mesh)
    end

    # re-order dofs
    A_inversion = A_inversion[mesh.dofs.p_inversion, mesh.dofs.p_inversion]
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

    # background state ∂z(b) = N²
    N² = 1.

    # initial condition: b = N²z, t = 0
    u = interpolate_everywhere(0, mesh.spaces.X_trial[1])
    v = interpolate_everywhere(0, mesh.spaces.X_trial[2])
    w = interpolate_everywhere(0, mesh.spaces.X_trial[3])
    p = interpolate_everywhere(0, mesh.spaces.X_trial[4]) 
    b = interpolate_everywhere(0, mesh.spaces.B_trial)
    t = 0.
    state = State(u, v, w, p, b, t) 

    # build evolution matrices and test against saved matrices
    θ = Δt/2 * ε^2 / μϱ
    A_diff_fname = @sprintf("test/data/A_diff_%s_%e_%e_%e.jld2", dim, h, θ, α)
    A_adv_fname = @sprintf("test/data/A_adv_%s_%e.jld2", dim, h)
    if !isfile(A_diff_fname) || !isfile(A_adv_fname)
        @warn "A_diff or A_adv file not found, generating..."
        A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ, N², Δt; 
                                            A_adv_ofile=A_adv_fname, A_diff_ofile=A_diff_fname)
    else
        A_adv, A_diff, B_diff, b_diff = build_evolution_matrices(mesh, params, κ, N², Δt) 
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

    # Krylov solver for evolution
    VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}
    tol = 1e-6
    itmax = 0
    solver_evolution = CgSolver(mesh.dofs.nb, mesh.dofs.nb, VT)
    solver_evolution.x .= on_architecture(arch, copy(state.b)[mesh.dofs.p_b])

    # evolution functions
    b_half = interpolate_everywhere(0, mesh.spaces.B_trial)
    function evolve_adv!(inversion_toolkit, solver_evolution, state)
        # determine architecture
        arch = architecture(solver_evolution.x)

        # unpack
        p_b = mesh.dofs.p_b
        inv_p_b = mesh.dofs.inv_p_b
        B_test = mesh.spaces.B_test
        dΩ = mesh.dΩ

        # half step
        l_half(d) = ∫( b*d - Δt/2*(u*∂x(b) + v*∂y(b) + w*(N² + ∂z(b)))*d )dΩ
        RHS = on_architecture(arch, assemble_vector(l_half, B_test)[p_b])
        Krylov.solve!(solver_evolution, A_adv, RHS, solver_evolution.x, M=P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)

        # update b_half
        b_half.free_values .= on_architecture(CPU(), solver_evolution.x[inv_p_b])

        # invert with b_half
        invert!(inversion_toolkit, b_half)
        set_state!(state, mesh, inversion_toolkit)

        # full step
        l_full(d) = ∫( b*d - Δt*(u*∂x(b_half) + v*∂y(b_half) + w*(N² + ∂z(b_half)))*d )dΩ
        RHS = on_architecture(arch, assemble_vector(l_full, B_test)[p_b])
        Krylov.solve!(solver_evolution, A_adv, RHS, solver_evolution.x, M=P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)

        # update state
        state.b .= on_architecture(CPU(), solver_evolution.x[inv_p_b])

        return inversion_toolkit, solver_evolution
    end
    function evolve_diff!(solver_evolution, state)
        # determine architecture
        arch = architecture(solver_evolution.x)

        # make copy of b on arch
        b_arch = on_architecture(arch, state.b)

        # compute RHS
        RHS = B_diff*b_arch + b_diff

        # solve
        Krylov.solve!(solver_evolution, A_diff, RHS, solver_evolution.x, M=P_diff, atol=tol, rtol=tol, verbose=0, itmax=itmax)

        # update state
        state.b .= on_architecture(CPU(), solver_evolution.x[mesh.dofs.inv_p_b])

        return solver_evolution
    end

    # solve function
    function solve!(state, inversion_toolkit, solver_evolution, i_step, n_steps)
        @showprogress for i ∈ i_step:n_steps
            # advection step
            evolve_adv!(inversion_toolkit, solver_evolution, state)

            # diffusion step
            evolve_diff!(solver_evolution, state)

            # invert
            invert!(inversion_toolkit, b)
            set_state!(state, mesh, inversion_toolkit)

            # time
            state.t += Δt

            # average stratification
            if mod(i, n_steps ÷ 100) == 0
                @info @sprintf("average ∂z(b) = %1.5e", sum(∫(N² + ∂z(b))mesh.dΩ)/sum(∫(1)mesh.dΩ))
            end
        end
        return state, inversion_toolkit, solver_evolution
    end

    # run
    i_step = Int64(round(t/Δt)) + 1
    n_steps = Int64(round(T/Δt))
    solve!(state, inversion_toolkit, solver_evolution, i_step, n_steps)

    # plot for sanity check
    sim_plots(dim, u, v, w, b, N², H, state.t, 0)

    # compare state with data
    datafile = @sprintf("test/data/evolution_%s.jld2", dim)
    if !isfile(datafile)
        @warn "Data file not found, saving state..."
        save(state; ofile=datafile)
    else
        state_data = load_state(datafile)
        @test isapprox(state.u, state_data.u, rtol=1e-2)
        @test isapprox(state.v, state_data.v, rtol=1e-2)
        @test isapprox(state.w, state_data.w, rtol=1e-2)
        @test isapprox(state.p, state_data.p, rtol=1e-2)
        @test isapprox(state.b, state_data.b, rtol=1e-2)
    end
end

@testset "Evolution Tests" begin
    # @testset "2D CPU" begin
    #     coarse_evolution(TwoD(), CPU())
    # end
    # @testset "2D GPU" begin
    #     coarse_evolution(TwoD(), GPU())
    # end
    @testset "3D CPU" begin
        coarse_evolution(ThreeD(), CPU())
    end
    # @testset "3D GPU" begin
    #     coarse_evolution(ThreeD(), GPU())
    # end
end