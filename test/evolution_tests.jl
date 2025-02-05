using Test
using nuPGCM
using Gridap, GridapGmsh
using IncompleteLU, Krylov, LinearOperators, CuthillMcKee
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra, Statistics
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function coarse_evolution(dim::AbstractDimension, arch::AbstractArchitecture)
    # tolerance and max iterations for iterative solvers
    tol = 1e-6
    itmax = 0

    # Vector type 
    VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}

    # coarse model
    h = 0.1
    model = GmshDiscreteModel(@sprintf("meshes/bowl%s_%0.2f.msh", dim, h))

    # FE spaces
    X, Y, B, D = setup_FESpaces(model)
    Ux, Uy, Uz, P = unpack_spaces(X)
    nx = Ux.space.nfree
    ny = Uy.space.nfree
    nz = Uz.space.nfree
    nu = nx + ny + nz
    np = P.space.space.nfree
    nb = B.space.nfree
    N = nu + np - 1

    # triangulation and integration measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)

    # depth
    H(x) = 1 - x[1]^2 - x[2]^2

    # forcing
    ν(x) = 1
    κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

    # params
    ε² = 1e-2
    γ = 1/4
    f₀ = 1
    β = 0.5
    f(x) = f₀ + β*x[2]
    μϱ = 1e-4
    Δt = 1e-4*μϱ/ε²
    T = 5e-2*μϱ/ε²
    α = Δt/2*ε²/μϱ

    # assemble LHS inversion and test against saved matrix
    LHS_inversion_fname = @sprintf("test/data/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, h, ε², γ, f₀, β)
    if !isfile(LHS_inversion_fname)
        @warn "LHS_inversion file not found, generating..."
        LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
    else
        # just read inversion matrix instead of building and testing it; this is already tested in inversion_tests.jl
        LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
    end

    # inversion RHS
    RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

    # preconditioner
    if typeof(arch) == CPU
        P_inversion = lu(LHS_inversion)
        ldiv_P_inversion = true
    else
        P_inversion = Diagonal(on_architecture(arch, 1/h^dim.n*ones(N)))
        ldiv_P_inversion = false
    end

    # put on GPU, if needed
    LHS_inversion = on_architecture(arch, LHS_inversion)
    RHS_inversion = on_architecture(arch, RHS_inversion)

    # Krylov solver for inversion
    solver_inversion = GmresSolver(N, N, 20, VT)
    solver_inversion.x .= 0.

    # inversion functions
    function invert!(arch::AbstractArchitecture, solver, b)
        b_arch = on_architecture(arch, b.free_values)
        if typeof(arch) == GPU
            RHS = [CUDA.zeros(nx); CUDA.zeros(ny); RHS_inversion*b_arch; CUDA.zeros(np-1)]
        else
            RHS = [zeros(nx); zeros(ny); RHS_inversion*b_arch; zeros(np-1)]
        end
        Krylov.solve!(solver, LHS_inversion, RHS, solver.x, M=P_inversion, ldiv=ldiv_P_inversion,
                    atol=tol, rtol=tol, verbose=0, itmax=itmax, restart=true,
                    history=true)
        return solver
    end
    function update_u_p!(ux, uy, uz, p, solver)
        sol = on_architecture(CPU(), solver.x[inv_perm_inversion])
        ux.free_values .= sol[1:nx]
        uy.free_values .= sol[nx+1:nx+ny]
        uz.free_values .= sol[nx+ny+1:nx+ny+nz]
        p = FEFunction(P, sol[nx+ny+nz+1:end])
        return ux, uy, uz, p
    end

    # background state \partial_z b = N^2
    N² = 1.

    # initial condition: b = N^2 z, t = 0
    b = interpolate_everywhere(0, B)
    t = 0.
    ux = interpolate_everywhere(0, Ux)
    uy = interpolate_everywhere(0, Uy)
    uz = interpolate_everywhere(0, Uz)
    p  = interpolate_everywhere(0, P) 

    # assemble evolution matrices and test against saved matrices
    LHS_diff_fname = @sprintf("test/data/LHS_diff_%s_%e_%e_%e.h5", dim, h, α, γ)
    LHS_adv_fname = @sprintf("test/data/LHS_adv_%s_%e.h5", dim, h)
    if !isfile(LHS_diff_fname) || !isfile(LHS_adv_fname)
        @warn "LHS_diff or LHS_adv file not found, generating..."
        LHS_adv, LHS_diff, perm_b, inv_perm_b = assemble_LHS_adv_diff(CPU(), α, γ, κ, B, D, dΩ; fname_adv=LHS_adv_fname, fname_diff=LHS_diff_fname)
    else
        LHS_adv, LHS_diff, perm_b, inv_perm_b = assemble_LHS_adv_diff(CPU(), α, γ, κ, B, D, dΩ; fname_adv="LHS_adv_temp.h5", fname_diff="LHS_diff_temp.h5")
        @test LHS_adv ≈ read_sparse_matrix(LHS_adv_fname)[1]
        @test LHS_diff ≈ read_sparse_matrix(LHS_diff_fname)[1]
    end

    # diffusion RHS matrix and vector
    RHS_diff, rhs_diff = assemble_RHS_diff(perm_b, α, γ, κ, N², B, D, dΩ)

    # preconditioners
    if typeof(arch) == CPU
        P_diff = lu(LHS_diff)
        P_adv  = lu(LHS_adv)
        ldiv_P_diff = true
        ldiv_P_adv  = true
    else
        P_diff = Diagonal(on_architecture(arch, Vector(1 ./ diag(LHS_diff))))
        P_adv  = Diagonal(on_architecture(arch, Vector(1 ./ diag(LHS_adv))))
        ldiv_P_diff = false
        ldiv_P_adv  = false
    end

    # put on GPU, if needed
    LHS_diff = on_architecture(arch, LHS_diff)
    RHS_diff = on_architecture(arch, RHS_diff)
    rhs_diff = on_architecture(arch, rhs_diff)
    LHS_adv = on_architecture(arch, LHS_adv)

    # Krylov solver for evolution
    solver_evolution = CgSolver(nb, nb, VT)
    solver_evolution.x .= on_architecture(arch, copy(b.free_values)[perm_b])

    # evolution functions
    b_half = interpolate_everywhere(0, B)
    function evolve_adv!(arch::AbstractArchitecture, solver_inversion, solver_evolution, ux, uy, uz, p, b)
        # half step
        l_half(d) = ∫( b*d - Δt/2*(ux*∂x(b) + uy*∂y(b) + uz*(N² + ∂z(b)))*d )dΩ
        RHS = on_architecture(arch, assemble_vector(l_half, D)[perm_b])
        Krylov.solve!(solver_evolution, LHS_adv, RHS, solver_evolution.x, M=P_adv, ldiv=ldiv_P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)

        # u, v, w, p, b at half step
        update_b!(b_half, solver_evolution)
        solver_inversion = invert!(arch, solver_inversion, b_half)
        ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

        # full step
        l_full(d) = ∫( b*d - Δt*(ux*∂x(b_half) + uy*∂y(b_half) + uz*(N² + ∂z(b_half)))*d )dΩ
        RHS = on_architecture(arch, assemble_vector(l_full, D)[perm_b])
        Krylov.solve!(solver_evolution, LHS_adv, RHS, solver_evolution.x, M=P_adv, ldiv=ldiv_P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)

        return solver_inversion, solver_evolution
    end
    function evolve_diff!(arch::AbstractArchitecture, solver, b)
        b_arch= on_architecture(arch, b.free_values)
        RHS = RHS_diff*b_arch + rhs_diff
        Krylov.solve!(solver, LHS_diff, RHS, solver.x, M=P_diff, ldiv=ldiv_P_diff, atol=tol, rtol=tol, verbose=0, itmax=itmax)
        return solver
    end
    function update_b!(b, solver)
        b.free_values .= on_architecture(CPU(), solver.x[inv_perm_b])
        return b
    end

    # solve function
    function solve!(arch::AbstractArchitecture, ux, uy, uz, p, b, t, solver_inversion, solver_evolution, i_step, n_steps)
        for i ∈ i_step:n_steps
            # advection step
            solver_inversion, solver_evolution = evolve_adv!(arch, solver_inversion, solver_evolution, ux, uy, uz, p, b)
            b = update_b!(b, solver_evolution)

            # diffusion step
            solver_evolution = evolve_diff!(arch, solver_evolution, b)
            b = update_b!(b, solver_evolution)

            # invert
            solver_inversion = invert!(arch, solver_inversion, b)
            ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

            # time
            t += Δt
        end
        return ux, uy, uz, p, b
    end

    # run
    i_step = Int64(round(t/Δt)) + 1
    n_steps = Int64(round(T/Δt))
    ux, uy, uz, p, b = solve!(arch, ux, uy, uz, p, b, t, solver_inversion, solver_evolution, i_step, n_steps)

    # # plot for sanity check
    # sim_plots(dim, ux, uy, uz, b, N², H, t, 0, "test")

    # FIRST TIME ONLY: save state
    # save_state(ux, uy, uz, p, b, t; fname=@sprintf("test/data/evolution_%s.h5", dim))

    # compare state with data
    datafile = @sprintf("test/data/evolution_%s.h5", dim)
    if !isfile(datafile)
        @warn "Data file not found, saving state..."
        save_state(ux, uy, uz, p, b, t; fname=@sprintf("test/data/evolution_%s.h5", dim))
    else
        ux_data, uy_data, uz_data, p_data, b_data, t_data = load_state(@sprintf("test/data/evolution_%s.h5", dim))
        @test isapprox(ux.free_values, ux_data, rtol=1e-2)
        @test isapprox(uy.free_values, uy_data, rtol=1e-2)
        @test isapprox(uz.free_values, uz_data, rtol=1e-2)
        @test isapprox(p.free_values,  p_data,  rtol=1e-2)
        @test isapprox(b.free_values,  b_data,  rtol=1e-2)
    end

    # remove temporary files
    rm("LHS_adv_temp.h5", force=true)
    rm("LHS_diff_temp.h5", force=true)
end

@testset "Evolution Tests" begin
    @testset "2D CPU" begin
        coarse_evolution(TwoD(), CPU())
    end
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