using nuPGCM
using Statistics
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

# choose architecture
arch = CPU()
# arch = GPU()

# tolerance and max iterations for iterative solvers
tol = 1e-5
itmax = 0

# Vector type 
VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}

# depth
H(x) = 1 - x[1]^2 - x[2]^2

# forcing
ν(x) = 1

# params
ε² = 1e-4
γ = 1/4
f₀ = 1
β = 0
f(x) = f₀ + β*x[2]

function compute_error(dim::AbstractDimension, hres; showplots=false)
    # model
    # model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_%0.2f.msh", dim, hres))
    model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_exp.msh", dim))

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

    # mesh resolution
    m = Mesh(model)
    hs = [norm(m.p[m.t[i, j], :] - m.p[m.t[i, mod1(j+1, dim.n+1)], :]) for i ∈ axes(m.t, 1), j ∈ 1:dim.n+1]
    h = mean(hs)
    @printf("mean(h) = %e\n", mean(hs))

    # triangulation and integration measure
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 4)

    # filenames for LHS matrices
    # LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, hres, ε², γ, f₀, β)
    LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%s_exp_%e_%e_%e_%e.h5", dim, ε², γ, f₀, β)

    # inversion LHS
    if isfile(LHS_inversion_fname)
        LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
    else
        LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, dim, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
    end

    # inversion RHS
    RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

    # preconditioner
    if typeof(dim) == TwoD
        P_inversion = Diagonal(1/h^2*ones(N))
    else
        P_inversion = Diagonal(1/h^3*ones(N))
    end

    # put on GPU, if needed
    LHS_inversion = on_architecture(arch, LHS_inversion)
    RHS_inversion = on_architecture(arch, RHS_inversion)
    P_inversion = Diagonal(on_architecture(arch, diag(P_inversion)))

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

    # b = z should have no flow
    b  = interpolate_everywhere(x->x[3], B)
    ux = interpolate_everywhere(0, Ux)
    uy = interpolate_everywhere(0, Uy)
    uz = interpolate_everywhere(0, Uz)
    p  = interpolate_everywhere(0, P)

    # invert
    if typeof(arch) == GPU
        solver_inversion = invert!(arch, solver_inversion, b)
        ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)
    else
        RHS = [zeros(nx); zeros(ny); RHS_inversion*b.free_values; zeros(np-1)]
        sol = LHS_inversion \ RHS
        sol = sol[inv_perm_inversion]
        ux.free_values .= sol[1:nx]
        uy.free_values .= sol[nx+1:nx+ny]
        uz.free_values .= sol[nx+ny+1:nx+ny+nz]
        p = FEFunction(P, sol[nx+ny+nz+1:end])
    end

    # compute error
    ∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
    ∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
    ∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)
    eu_L2 = sqrt(sum( ∫( ux*ux + uy*uy + uz*uz )*dΩ ))
    eu_H1 = sqrt(sum( ∫( ux*ux + uy*uy + uz*uz + 
                            ∂x(ux)*∂x(ux) + ∂y(ux)*∂y(ux) + ∂z(ux)*∂z(ux) +
                            ∂x(uy)*∂x(uy) + ∂y(uy)*∂y(uy) + ∂z(uy)*∂z(uy) +
                            ∂x(uz)*∂x(uz) + ∂y(uz)*∂y(uz) + ∂z(uz)*∂z(uz) 
                            )*dΩ ))
    p0 = interpolate_everywhere(x->x[3]^2/2, P) # since P is a zero-mean space, Gridap will automatically subtract the mean
    ep_L2 = sqrt(sum( ∫( (p - p0)*(p - p0) )*dΩ ))
    @printf("    h = %e\n", h)
    @printf(" |u|₂ = %e\n", eu_L2)
    @printf(" |u|₁ = %e\n", eu_H1)
    @printf(" |p|₀ = %e\n", ep_L2)
    @printf("error = %e\n", eu_H1 + ep_L2)

    if showplots
        b.free_values .= 0
        plot_slice(ux*ux + uy*uy + uz*uz, 
                b; y=0, cb_label=L"$|\mathbf{u}|^2$", 
                fname=@sprintf("%s/images/u_L2_err_%1.2f.png", out_folder, hres))
        plot_slice(ux*ux + uy*uy + uz*uz + 
                ∂x(ux)*∂x(ux) + ∂y(ux)*∂y(ux) + ∂z(ux)*∂z(ux) +     
                ∂x(uy)*∂x(uy) + ∂y(uy)*∂y(uy) + ∂z(uy)*∂z(uy) +     
                ∂x(uz)*∂x(uz) + ∂y(uz)*∂y(uz) + ∂z(uz)*∂z(uz),
                b; y=0, cb_label=L"$|\mathbf{u}|^2 + |\nabla\mathbf{u}|^2$", 
                fname=@sprintf("%s/images/u_H1_err_%1.2f.png", out_folder, hres))
        plot_slice((p - p0)*(p - p0), 
                b; y=0, cb_label=L"$|p - p_a|^2$", fname=@sprintf("%s/images/p_err_%1.2f.png", out_folder, hres))
    end

    return h, eu_L2, eu_H1, ep_L2
end

function plot_convergence_2D()
    # hs = [0.01, 0.02, 0.05]
    # hs = [7.96667e-03, 1.57642e-02, 3.80418e-02]
    hs = [9.96820e-03, 1.98866e-02, 4.91265e-02]

    errs = [
        [6.22111e-07, 3.42784e-06, 3.35299e-05], 

        [6.22111e-07, 3.42785e-06, 3.35303e-05], 

        [1.52113e-06, 8.14764e-06, 7.94241e-05], 
        
        [1.15721e-04, 6.07887e-04, 5.90033e-03],
        [1.50023e-04, 6.10646e-04, 5.90069e-03],
        [9.52791e-04, 7.61190e-04, 5.93063e-03],
        # [3.35898e-04, 6.14795e-04, 5.90047e-03],
        # [1.19563e-04, 6.14795e-04, 5.90047e-03],

        [1.15283e-02, 6.01363e-02, 5.67232e-01], 
        [1.16039e-02, 6.27308e-02, 5.78880e-01],
        [1.50846e-02, 6.27308e-02, 5.78880e-01],
        # [1.16003e-02, 6.27349e-02, 5.78861e-01],
        # [1.48924e-02, 6.27349e-02, 5.78861e-01]
    ]
    labels = [
        L"\varepsilon^2 = 10^{0},  \; f = 0, \; \gamma = 1",

        L"\varepsilon^2 = 10^{0},  \; f = 1, \; \gamma = 1",

        L"\varepsilon^2 = 10^{0},  \; f = 1, \; \gamma = 1/4",

        L"\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
        # L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-8}$",
        # L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-9}$",

        L"\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
    ]

    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(true)
    ax.spines["right"].set_visible(true)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
    ax.set_xlim(9e-3, 6e-2)
    ax.set_ylim(1e-7, 1e0)
    ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
    ax.set_axisbelow(true)
    for i ∈ eachindex(errs)
        ax.plot(hs, errs[i], "o-", label=labels[i])
    end
    ax.plot(hs, errs[1][2]/hs[2]^2*hs.^2, "k--", label=L"$O(h^2)$")
    ax.legend(loc=(1.05, 0.0))
    ax.set_title("2D Bowl (Gridap)")
    savefig(@sprintf("%s/images/convergence2D.png", out_folder))
    println(@sprintf("%s/images/convergence2D.png", out_folder))
    plt.close()
end

function plot_convergence_3D()
    hs = [9.62898e-03, 1.88410e-02, 4.45354e-02]

    errs = [
        [1.65234e-01, NaN, NaN],
        [1.65596e-01, 6.42308e-01, 3.66494e+00],
        [1.701087e-01, NaN, NaN],
        [2.101807e-01, NaN, NaN],
        [2.438967e-03, 2.211299e-02, 4.085575e-02],
        [NaN, 6.901556e-03, 4.039652e-02],
    ]
    labels = [
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-6}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-4}$",
        L"$\varepsilon^2 = 10^{-4}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-3}$",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-4}$",
        L"$\varepsilon^2 = 10^{-2}, \; f = 1, \; \gamma = 1/4$, tol$= 10^{-5}$",
    ]

    println((errs[2][1] - errs[1][1])/errs[1][1])
    println((errs[3][1] - errs[1][1])/errs[1][1])
    println((errs[4][1] - errs[1][1])/errs[1][1])

    fig, ax = plt.subplots(1, figsize=(3.2, 3.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(true)
    ax.spines["right"].set_visible(true)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1} + ||p - p_a||_{L^2}$")
    ax.set_xlim(7e-3, 5e-2)
    ax.set_ylim(1e-4, 1e1)
    ax.grid(true, which="both", color="k", alpha=0.5, linestyle=":", linewidth=0.25)
    ax.set_axisbelow(true)
    for i ∈ eachindex(errs)
        ax.plot(hs, errs[i], "o-", label=labels[i])
    end
    ax.plot(hs, errs[2][2]/hs[2]^2*hs.^2, "k--", label=L"$O(h^2)$")
    ax.legend(loc=(1.05, 0.0))
    ax.set_title("3D Bowl (Gridap)")
    savefig(@sprintf("%s/images/convergence3D.png", out_folder))
    println(@sprintf("%s/images/convergence3D.png", out_folder))
    plt.close()
end

# showplots = false
showplots = true

dim = TwoD()
# # dim = ThreeD()

h5, eu5_L2, eu5_H1, ep5_L2 = compute_error(dim, 0.05; showplots)
# h2, eu2_L2, eu2_H1, ep2_L2 = compute_error(dim, 0.02; showplots)
# h1, eu1_L2, eu1_H1, ep1_L2 = compute_error(dim, 0.01; showplots)

# @printf("[%1.5e, %1.5e, %1.5e]\n", h1, h2, h5)
# @printf("[%1.5e, %1.5e, %1.5e]\n", eu1_L2, eu2_L2, eu5_L2)
# @printf("[%1.5e, %1.5e, %1.5e]\n", eu1_H1, eu2_H1, eu5_H1)
# @printf("[%1.5e, %1.5e, %1.5e]\n", ep1_L2, ep2_L2, ep5_L2)
# @printf("[%1.5e, %1.5e, %1.5e]\n", eu1_H1 + ep1_L2, eu2_H1 + ep2_L2, eu5_H1 + ep5_L2)

# plot_convergence_2D()
# plot_convergence_3D()