using nuPGCM
using Gridap, GridapGmsh
using IncompleteLU, Krylov, LinearOperators, CuthillMcKee
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

out_folder = "../out"

function compute_error(hres)
    # choose dimensions
    dim = TwoD()
    # dim = ThreeD()

    # choose architecture
    arch = CPU()
    # arch = GPU()

    # tolerance and max iterations for iterative solvers
    tol = 1e-8
    @printf("tol = %.1e\n", tol)
    itmax = 0
    @printf("itmax = %d\n", itmax)

    # Vector type 
    VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}

    # model
    model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_%0.2f.msh", dim, hres))

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
    @printf("\nN = %d (%d + %d) ∼ 10^%d DOF\n", N, nu, np-1, floor(log10(N)))

    # triangulation and integration measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)

    # depth
    H(x) = 1 - x[1]^2 - x[2]^2

    # forcing
    ν(x) = 1

    # params
    ε² = 1
    γ = 1
    f₀ = 0
    β = 0
    f(x) = f₀ + β*x[2]
    println("\n---")
    println("Parameters:\n")
    @printf("ε² = %.1e (δ = %.1e, h = %.1e)\n", ε², √(2ε²), hres)
    @printf("f₀ = %.1e\n", f₀)
    @printf(" β = %.1e\n", β)
    @printf(" γ = %.1e\n", γ)
    println("---\n")

    # filenames for LHS matrices
    LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, hres, ε², γ, f₀, β)

    # inversion LHS
    # if isfile(LHS_inversion_fname)
    #     LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
    # else
    #     LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, dim, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
    # end
    LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, dim, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)

    # inversion RHS
    RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

    # preconditioner
    P_inversion = I
    # P_inversion = Diagonal(1/hres^3*ones(N))

    # put on GPU, if needed
    LHS_inversion = on_architecture(arch, LHS_inversion)
    RHS_inversion = on_architecture(arch, RHS_inversion)
    # P_inversion = Diagonal(on_architecture(arch, diag(P_inversion)))
    if typeof(arch) == GPU
        CUDA.memory_status()
        println()
    end

    # Krylov solver for inversion
    solver_inversion = GmresSolver(N, N, 20, VT)
    solver_inversion.x .= on_architecture(arch, zeros(N))

    # inversion functions
    function invert!(arch::AbstractArchitecture, solver, b)
        b_arch = on_architecture(arch, b.free_values)
        if typeof(arch) == GPU
            RHS = [CUDA.zeros(nx); CUDA.zeros(ny); RHS_inversion*b_arch; CUDA.zeros(np-1)]
        else
            RHS = [zeros(nx); zeros(ny); RHS_inversion*b_arch; zeros(np-1)]
        end
        Krylov.solve!(solver, LHS_inversion, RHS, solver.x, M=P_inversion, 
                    atol=tol, rtol=tol, verbose=0, itmax=itmax, restart=true)
        @printf("inversion GMRES: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
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

    flush(stdout)
    flush(stderr)

    # b = z should have no flow
    b  = interpolate_everywhere(x->x[3], B)
    ux = interpolate_everywhere(0, Ux)
    uy = interpolate_everywhere(0, Uy)
    uz = interpolate_everywhere(0, Uz)
    p  = interpolate_everywhere(0, P)

    # # invert
    # solver_inversion = invert!(arch, solver_inversion, b)
    # ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

    # invert exact
    RHS = [zeros(nx); zeros(ny); RHS_inversion*b.free_values; zeros(np-1)]
    sol = LHS_inversion \ RHS
    sol = sol[inv_perm_inversion]
    ux.free_values .= sol[1:nx]
    uy.free_values .= sol[nx+1:nx+ny]
    uz.free_values .= sol[nx+ny+1:nx+ny+nz]
    p = FEFunction(P, sol[nx+ny+nz+1:end])

    # compute error
    ∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
    ∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
    ∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)
    p0 = interpolate_everywhere(x->x[3]^2/2, P) # since P is a zero-mean space, Gridap will automatically subtract the mean
    ep_L2 = sqrt(sum( ∫( (p - p0)*(p - p0) )*dΩ ))
    eu_H1 = sqrt(sum( ∫( ux*ux + uy*uy + uz*uz + 
                         ∂x(ux)*∂x(ux) + ∂y(ux)*∂y(ux) + ∂z(ux)*∂z(ux) +
                         ∂x(uy)*∂x(uy) + ∂y(uy)*∂y(uy) + ∂z(uy)*∂z(uy) +
                         ∂x(uz)*∂x(uz) + ∂y(uz)*∂y(uz) + ∂z(uz)*∂z(uz) 
                         )*dΩ ))
    @printf("error = %e\n", ep_L2 + eu_H1)
    h = 1/sqrt(np)

    # plot error
    b.free_values .= 0
    plot_slice(ux*ux + uy*uy + uz*uz + 
               ∂x(ux)*∂x(ux) + ∂y(ux)*∂y(ux) + ∂z(ux)*∂z(ux) +     
               ∂x(uy)*∂x(uy) + ∂y(uy)*∂y(uy) + ∂z(uy)*∂z(uy) +     
               ∂x(uz)*∂x(uz) + ∂y(uz)*∂y(uz) + ∂z(uz)*∂z(uz),
               b; y=0, cb_label=L"$|\mathbf{u}|^2 + |\nabla\mathbf{u}|^2$", 
               fname=@sprintf("%s/images/u_H1_err_%1.2f.png", out_folder, hres))
    plot_slice((p - p0)*(p - p0), b; y=0, cb_label=L"$|p - p_a|^2$", fname=@sprintf("%s/images/p_err_%1.2f.png", out_folder, hres))
    return h, ep_L2 + eu_H1
end

function plot_convergence()
    # hs = [0.01, 0.02, 0.05]
    hs = [7.96667e-03, 1.57642e-02, 3.80418e-02]

    err1 = [5.785239e-07, 3.201639e-06, 3.118399e-05]
    err2 = [6.221112e-07, 3.427847e-06, 3.353035e-05]

    # err_f0 = [5.785239e-07, 3.201639e-06, 3.118399e-05]
    # err_f0_diri = [5.80427e-07, 3.21156e-06, 3.12326e-05]
    # err_L2s = [2.914997e-04, 1.575368e-03, 9.549553e-03]
    # err_H1s = [8.961786e-03, 4.652045e-02, 4.170913e-01]
    # err_L2s_direct = [1.863660e-05, 1.986254e-04, 4.864734e-03]
    # err_H1s_direct = [8.915641e-03, 4.635080e-02, 4.161722e-01]
    # err_L2s_γ1_ε1 = [4.172885e-10, 4.406407e-09, 1.061986e-07]
    # err_H1s_γ1_ε1 = [2.090514e-07, 1.107825e-06, 1.053157e-05]
    # err_L2s_f0 = [4.172884e-10, 4.406407e-09, 1.061986e-07]
    # err_H1s_f0 = [2.090514e-07, 1.107825e-06, 1.053157e-05]

    # err = err_f0_diri

    fig, ax = plt.subplots(1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
    ax.plot(hs, err1, "o-")
    ax.plot(hs, err2, "o-")
    h = [hs[1], hs[end]]
    ax.plot(h, err1[end]/h[end]^2*h.^2, "k--", alpha=0.5, label=L"$O(h^2)$")
    ax.plot(h, err1[end]/h[end]^3*h.^3, "k--", alpha=1.0, label=L"$O(h^3)$")
    # ax.legend(ncol=3, loc=(0.05, 1.05))
    ax.legend()
    ax.set_title("Bowl (Gridap)")
    ax.set_xlim(5e-3, 2e-1)
    ax.set_ylim(1e-7, 1e-2)
    savefig(@sprintf("%s/images/convergence.png", out_folder))
    println(@sprintf("%s/images/convergence.png", out_folder))
    plt.close()
end

h5, err5 = compute_error(0.05)
# h2, err2 = compute_error(0.02)
# h1, err1 = compute_error(0.01)
# @printf("[%1.5e, %1.5e, %1.5e]\n", h1, h2, h5)
# @printf("[%1.5e, %1.5e, %1.5e]\n", err1, err2, err5)

# plot_convergence()