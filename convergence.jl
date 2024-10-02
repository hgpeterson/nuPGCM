# using NonhydroPG
# using Gridap, GridapGmsh
# using IncompleteLU, Krylov, LinearOperators, CuthillMcKee
# using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
# using SparseArrays, LinearAlgebra
# using Printf
# using PyPlot

# pygui(false)
# plt.style.use("plots.mplstyle")
# plt.close("all")

# out_folder = "out"

# if !isdir(out_folder)
#     println("creating folder: ", out_folder)
#     mkdir(out_folder)
# end
# if !isdir("$out_folder/images")
#     println("creating subfolder: ", out_folder, "/images")
#     mkdir("$out_folder/images")
# end
# if !isdir("$out_folder/data")
#     println("creating subfolder: ", out_folder, "/data")
#     mkdir("$out_folder/data")
# end
# flush(stdout)
# flush(stderr)

# # choose dimensions
# dim = TwoD()
# # dim = ThreeD()

# # choose architecture
# # arch = CPU()
# arch = GPU()

# # tolerance and max iterations for iterative solvers
# tol = 1e-8
# @printf("tol = %.1e\n", tol)
# itmax = 0
# @printf("itmax = %d\n", itmax)

# # Vector type 
# VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}

# # model
# hres = 0.01
# model = GmshDiscreteModel(@sprintf("meshes/bowl%s_%0.2f.msh", dim, hres))

# # full grid
# m = Mesh(model)

# # surface grid
# m_sfc = Mesh(model, "sfc")

# # mesh res
# hs = [norm(m.p[m.t[i, j], :] - m.p[m.t[i, mod1(j+1, dim.n+1)], :]) for i ∈ axes(m.t, 1), j ∈ 1:dim.n+1]
# hmin = minimum(hs)
# hmax = maximum(hs)

# # FE spaces
# X, Y, B, D = setup_FESpaces(model)
# Ux, Uy, Uz, P = unpack_spaces(X)
# nx = Ux.space.nfree
# ny = Uy.space.nfree
# nz = Uz.space.nfree
# nu = nx + ny + nz
# np = P.space.space.nfree
# nb = B.space.nfree
# N = nu + np - 1
# @printf("\nN = %d (%d + %d) ∼ 10^%d DOF\n", N, nu, np-1, floor(log10(N)))

# # triangulation and integration measure
# Ω = Triangulation(model)
# dΩ = Measure(Ω, 4)

# # depth
# H(x) = 1 - x[1]^2 - x[2]^2

# # forcing
# ν(x) = 1
# κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

# # params
# ε² = 1e-4
# γ = 1/4
# f₀ = 1
# β = 0
# f(x) = f₀ + β*x[2]
# println("\n---")
# println("Parameters:\n")
# @printf("ε² = %.1e (δ = %.1e, %.1e ≤ h ≤ %.1e)\n", ε², √(2ε²), hmin, hmax)
# @printf("f₀ = %.1e\n", f₀)
# @printf(" β = %.1e\n", β)
# @printf(" γ = %.1e\n", γ)
# println("---\n")

# # filenames for LHS matrices
# LHS_inversion_fname = @sprintf("matrices/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, hres, ε², γ, f₀, β)

# # inversion LHS
# if isfile(LHS_inversion_fname)
#     LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
# else
#     LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, dim, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
# end

# # inversion RHS
# RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

# # preconditioner
# P_inversion = I
# # P_inversion = Diagonal(1/hres^3*ones(N))

# # put on GPU, if needed
# LHS_inversion = on_architecture(arch, LHS_inversion)
# RHS_inversion = on_architecture(arch, RHS_inversion)
# # P_inversion = Diagonal(on_architecture(arch, diag(P_inversion)))
# if typeof(arch) == GPU
#     CUDA.memory_status()
#     println()
# end

# # Krylov solver for inversion
# solver_inversion = GmresSolver(N, N, 20, VT)
# solver_inversion.x .= on_architecture(arch, zeros(N))

# # inversion functions
# function invert!(arch::AbstractArchitecture, solver, b)
#     b_arch = on_architecture(arch, b.free_values)
#     if typeof(arch) == GPU
#         RHS = [CUDA.zeros(nx); CUDA.zeros(ny); RHS_inversion*b_arch; CUDA.zeros(np-1)]
#     else
#         RHS = [zeros(nx); zeros(ny); RHS_inversion*b_arch; zeros(np-1)]
#     end
#     Krylov.solve!(solver, LHS_inversion, RHS, solver.x, M=P_inversion, 
#                   atol=tol, rtol=tol, verbose=0, itmax=itmax, restart=true)
#     @printf("inversion GMRES: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
#     return solver
# end
# function update_u_p!(ux, uy, uz, p, solver)
#     sol = on_architecture(CPU(), solver.x[inv_perm_inversion])
#     ux.free_values .= sol[1:nx]
#     uy.free_values .= sol[nx+1:nx+ny]
#     uz.free_values .= sol[nx+ny+1:nx+ny+nz]
#     p = FEFunction(P, sol[nx+ny+nz+1:end])
#     return ux, uy, uz, p
# end

# flush(stdout)
# flush(stderr)

# # b = z should have no flow
# b  = interpolate_everywhere(x->x[3], B)
# ux = interpolate_everywhere(0, Ux)
# uy = interpolate_everywhere(0, Uy)
# uz = interpolate_everywhere(0, Uz)
# p  = interpolate_everywhere(0, P)

# # invert
# solver_inversion = invert!(arch, solver_inversion, b)
# ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

# # compute error
# ∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
# ∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
# ∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)
# err_L2 = sqrt(sum( ∫( ux*ux + uy*uy + uz*uz )*dΩ ))
# err_H1 = sqrt(sum( ∫( ux*ux + uy*uy + uz*uz + ∂x(ux)*∂x(ux) + ∂y(uy)*∂y(uy) + ∂z(uz)*∂z(uz) )*dΩ ))
# @printf("L2 error = %e\n", err_L2)
# @printf("H1 error = %e\n", err_H1)

# function slope(err1, err2, h1, h2)
#     return log10(err1/err2)/log10(h1/h2)
# end

function plot_convergence()
    hs = [0.01, 0.02, 0.05]
    err_L2s = [2.914997e-04, 1.575368e-03, 9.549553e-03]
    err_H1s = [8.961786e-03, 4.652045e-02, 4.170913e-01]
    fig, ax = plt.subplots(1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||\mathbf{u}||_{H^1}$")
    ax.plot(hs, err_H1s, "o-")
    h = range(0.01, 0.05, length=100)
    ax.plot(h, 3e2*h.^2, "--", label=L"$h^2$")
    ax.plot(h, 2e3*h.^3, "--", label=L"$h^3$")
    ax.legend()
    savefig(@sprintf("%s/images/convergence.png", out_folder))
    println(@sprintf("%s/images/convergence.png", out_folder))
    plt.close()
end

plot_convergence()