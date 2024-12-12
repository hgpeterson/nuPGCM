using nuPGCM
using Gridap, GridapGmsh
using IncompleteLU, Krylov, LinearOperators, CuthillMcKee
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra, Statistics
using JLD2
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

out_folder = "../out"

if !isdir(out_folder)
    println("creating folder: ", out_folder)
    mkdir(out_folder)
end
if !isdir("$out_folder/images")
    println("creating subfolder: ", out_folder, "/images")
    mkdir("$out_folder/images")
end
if !isdir("$out_folder/data")
    println("creating subfolder: ", out_folder, "/data")
    mkdir("$out_folder/data")
end
flush(stdout)
flush(stderr)

# choose dimensions
dim = TwoD()
# dim = ThreeD()

# choose architecture
arch = CPU()
# arch = GPU()

# tolerance and max iterations for iterative solvers
atol = 1e-6
rtol = sqrt(eps(Float64))
itmax = 0
@printf("atol = %.1e\n", atol)
@printf("rtol = %.1e\n", rtol)
@printf("itmax = %d\n\n", itmax)

# Vector type 
VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}

# model
hres = 0.01
model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_%0.2f.msh", dim, hres))

# full grid
m = Mesh(model)

# surface grid
m_sfc = Mesh(model, "sfc")

# mesh res
hs = [norm(m.p[m.t[i, j], :] - m.p[m.t[i, mod1(j+1, dim.n+1)], :]) for i ∈ axes(m.t, 1), j ∈ 1:dim.n+1]
hmin = minimum(hs)
hmax = maximum(hs)
h = mean(hs)
@printf("\n%.1e < h < %.1e (mean = %.1e)\n", hmin, hmax, h)

# FE spaces
X, Y, B, D = setup_FESpaces(model)
Ux, Uy, Uz, P = unpack_spaces(X)
Vx, Vy, Vz, Q = unpack_spaces(Y)
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
ε² = 1e-4
γ = 1
f₀ = 1
β = 0
f(x) = f₀ + β*x[2]
println("\n---")
println("Parameters:\n")
@printf("ε² = %.1e (δ = %.1e, %.1e ≤ h ≤ %.1e)\n", ε², √(2ε²), hmin, hmax)
@printf("f₀ = %.1e\n", f₀)
@printf(" β = %.1e\n", β)
@printf(" γ = %.1e\n", γ)
println("---\n")

# filenames for LHS matrices
LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, hres, ε², γ, f₀, β)

# inversion LHS
if isfile(LHS_inversion_fname)
    LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
else
    LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, dim, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
end
perm_u = perm_inversion[1:nu]
perm_p = perm_inversion[nu+1:end] .- nu

### preconditioners

# ldiv functions (Ay = v)
z = on_architecture(arch, zeros(nu))
function ldiv_chol!(y, P, v, z) 
    ldiv!(z, LowerTriangular(P), v)   # Forward substitution with L
    ldiv!(y, LowerTriangular(P)', z)  # Backward substitution with Lᴴ
    return y
end
function ldiv_chol!(y, P, v)
    y .= P \ v
    return y
end
function ldiv_lu!(y, P, v, z)
    ldiv!(z, UnitLowerTriangular(P), v)  # Forward substitution with L
    ldiv!(y, UpperTriangular(P), z)      # Backward substitution with U
    return y
end
function ldiv_lu!(y, L, U, v, z)
    ldiv!(z, L, v)  # Forward substitution with L
    ldiv!(y, U, z)  # Backward substitution with U
    return y
end

# blocks
dropzeros!(LHS_inversion)
A = LHS_inversion[1:nu, 1:nu]
# println("\nA is symmetric: ", issymmetric(A))
# println("A is posdef: ", isposdef(A), "\n")
C = LHS_inversion[nu+1:end, 1:nu]
CT = LHS_inversion[1:nu, nu+1:end]

# approximation of A matrix \sim -\nabla \cdot (K_\nu\nabla _) + fz \times _

# @time "factorize A" MA = lu(A)
# @time "factorize A" MA = ilu(A, τ=1e-3)

# a_ma((ux, uy, uz), (vx, vy, vz)) = ∫(# -uy*vx*f 
#                                      #+ ux*vy*f 
#                                      + γ*ε²*∂x(ux)*∂z(vx)*ν 
#                                      + γ*ε²*∂y(ux)*∂z(vx)*ν 
#                                      +   ε²*∂z(ux)*∂z(vx)*ν 
#                                      + γ*ε²*∂x(uy)*∂z(vy)*ν 
#                                      + γ*ε²*∂y(uy)*∂z(vy)*ν 
#                                      +   ε²*∂z(uy)*∂z(vy)*ν 
#                                      + γ^2*ε²*∂x(uz)*∂z(vz)*ν 
#                                      + γ^2*ε²*∂y(uz)*∂z(vz)*ν 
#                                      +   γ*ε²*∂z(uz)*∂z(vz)*ν )dΩ
# U = MultiFieldFESpace([Ux, Uy, Uz])
# V = MultiFieldFESpace([Vx, Vy, Vz])
# MA = assemble_matrix(a_ma, U, V) 
# MA = MA[perm_u, perm_u]
# @time "lu(MA)" MA = lu(MA)

# a_mu(u, v) = ∫( u*v*f )dΩ
# M_u = assemble_matrix(a_mu, Ux, Vx)
# M_u = M_u[perm_u[1:nx], perm_u[1:nx]]
# @assert norm(M_u - A[nx+1:nx+ny, 1:nx]) == 0.0
# @assert norm(M_u + A[1:nx, nx+1:nx+ny]) == 0.0
M_u = A[nx+1:nx+ny, 1:nx]
@assert norm(M_u + A[1:nx, nx+1:nx+ny]) == 0.0
M_u = lu(M_u)
K_u = A[1:nx, 1:nx]
@assert norm(K_u - A[nx+1:nx+ny, nx+1:nx+ny]) == 0.0
K_u = lu(K_u)
K_w = A[nx+ny+1:end, nx+ny+1:end]
K_w = lu(K_w)
function prod_MA!(y, v)
    # y[1:nx]        .=   M_u \ v[nx+1:nx+ny]
    # y[nx+1:nx+ny]  .= -(M_u \ v[1:nx])
    y[1:nx]        .= K_u \ v[1:nx]
    y[nx+1:nx+ny]  .= K_u \ v[nx+1:nx+ny]
    y[nx+ny+1:end] .= K_w \ v[nx+ny+1:end]
    return y
end
op_MA = LinearOperator(Float64, nu, nu, false, false, (y, v) -> prod_MA!(y, v))

# K_u = A[1:nx, 1:nx]
# K_u = lu(K_u)
# K_w = A[nx+ny+1:end, nx+ny+1:end]
# K_w = lu(K_w)
# M_f = A[nx+1:nx+ny, 1:nx]
# M_f = lu(M_f)
# op_u = LinearOperator(Float64, nx, nx, true, true, (y, v) -> ldiv!(y, K_u, v))
# op_v = LinearOperator(Float64, nx, nx, true, true, (y, v) -> ldiv!(y, K_u, v))
# op_w = LinearOperator(Float64, nz, nz, true, true, (y, v) -> ldiv!(y, K_w, v))

# MA_L = on_architecture(arch, sparse(MA.L + I))
# MA_L = UnitLowerTriangular(on_architecture(arch, MA.L))
# MA_U = on_architecture(arch, sparse(MA.U'))
# MA_U = UpperTriangular(on_architecture(arch, sparse(MA.U')))

# op_MA = LinearOperator(Float64, nu, nu, false, false, (y, v) -> ldiv!(y, MA, v))
# op_MA = LinearOperator(Float64, nu, nu, false, false, (y, v) -> ldiv_lu!(y, MA_L, MA_U, v, z))

# P_A = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
# A = on_architecture(arch, A)
# function mul_A!(y, v)
#     # z, stats = gmres(A, v, M=P_A, restart=true, memory=20, atol=atol, rtol=rtol, itmax=1000)
#     z, stats = gmres(A, v, restart=true, memory=20, atol=atol, rtol=rtol, itmax=1000)
#     y .= z
#     return y
# end
# op_MA = LinearOperator(Float64, nu, nu, true, true, (y, v) -> mul_A!(y, v))

# # pressure mass matrix
# a_mp(p, q) = ∫( p*q )dΩ
# M_p = assemble_matrix(a_mp, P, Q) 
# M_p = M_p[perm_p, perm_p]
# P_M_p = Diagonal(on_architecture(arch, Vector(1 ./ diag(M_p))))

# # pressure A matrix
# a_ap(p, q) = ∫( γ*ε²*∂x(p)*∂x(q)*ν + γ*ε²*∂y(p)*∂y(q)*ν + ε²*∂z(p)*∂z(q)*ν + p*q*f )dΩ
# A_p = assemble_matrix(a_ap, P, Q) 
# A_p = A_p[perm_p, perm_p]

# pressure K matrix over f
a_kp(p, q) = ∫( (∂x(p)*∂x(q) + ∂y(p)*∂y(q) + ∂z(p)*∂z(q))/f )dΩ
K_p = assemble_matrix(a_kp, P, Q) 
K_p = K_p[perm_p, perm_p]
P_K_p = lu(K_p)
# P_K_p = Diagonal(on_architecture(arch, Vector(1 ./ diag(K_p))))
# println("\nK_p is symmetric: ", issymmetric(K_p))
# println("K_p is posdef: ", isposdef(K_p), "\n")
# K_p = on_architecture(arch, K_p)

# # velocity mass matrix
# a_mu((ux, uy, uz), (vx, vy, vz)) = ∫( ux*vx + uy*vy + uz*vz )dΩ
# U = MultiFieldFESpace([Ux, Uy, Uz])
# V = MultiFieldFESpace([Vx, Vy, Vz])
# M_u = assemble_matrix(a_mu, U, V) 
# M_u = M_u[perm_u, perm_u]
# T_u = Diagonal(on_architecture(arch, Vector(1 ./ diag(M_u))))

# approximation of Schur complement S = C*A^{-1}*C^T
# solver = CgSolver(np-1, np-1, VT)
# solver.x .= 0
# L = C*T_u*CT
# L = C*T_u*C'
# println("\nL is symmetric: ", issymmetric(L))
# println("L is posdef: ", isposdef(L), "\n")
# # P_L = lu(L)
# P_L = Diagonal(on_architecture(arch, Vector(1 ./ diag(L))))
# L = on_architecture(arch, L)
# C = on_architecture(arch, C)
# CT = on_architecture(arch, CT)
# A = on_architecture(arch, A)
function MS_solve!(y, v)
    # Krylov.solve!(solver, M_p, ε²*v, M=P_M_p, ldiv=false)

    # Krylov.solve!(solver, M_p, v, M=P_M_p, ldiv=false)
    # solver.x .= A_p*solver.x
    # Krylov.solve!(solver, K_p, solver.x, M=P_K_p, ldiv=true)

    # z, stats = Krylov.cg(M_p, v, M=P_M_p, ldiv=false)
    # z, stats = Krylov.cg(K_p, A_p*z, M=P_K_p, ldiv=true)
    # y .= z

    # z1, stats = Krylov.cg(M_p, ε²*v, M=P_M_p, ldiv=false)
    # z2, stats = Krylov.cg(K_p,    v, M=P_K_p, ldiv=true)
    # y .= z1 + z2

    # v, stats = Krylov.cg(L, v, M=P_L)
    # # v .= P_L \ v
    # # v .= C*T_u*A*T_u*CT*v
    # v .= C*T_u*A*T_u*C'*v
    # v, stats = Krylov.cg(L, v, M=P_L)
    # # v .= P_L \ v
    # y .= v

    y .= P_K_p \ v
    # z, stats = Krylov.cg(K_p, v, M=P_K_p)
    # y .= z

    return y
end
op_MS = LinearOperator(Float64, np-1, np-1, true, true, (y, v) -> MS_solve!(y, v))

# full preconditioner = [MA BT; 0 -MS] -> inverse [MA^{-1} MA^{-1}*BT*MS^{-1}; 0 -MS^{-1}]
CT = on_architecture(arch, CT)
function mul_P!(y, v)
    y[1:nu] .= op_MA*(v[1:nu] + CT*op_MS*v[nu+1:end])
    y[nu+1:end] .= -op_MS*v[nu+1:end]
    return y
end
P_inversion = LinearOperator(Float64, N, N, false, false, (y, v) -> mul_P!(y, v))
ldiv_P_inversion = false

# a_lp(p, q) = ∫( ∂x(p)*∂x(q) + ∂y(p)*∂y(q) + ∂z(p)*∂z(q) )dΩ
# # a_lp(p, q) = ∫( ∂z(p)*∂z(q) )dΩ
# L_p = assemble_matrix(a_lp, P, Q) 
# L_p = L_p[perm_p, perm_p]
# MS = Diagonal(on_architecture(arch, Vector(1 ./ diag(M_p + L_p))))

# # linear operators of the form `LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)`
# # op1 = LinearOperator(Float64, nu, nu, true, true, (y, v) -> mul!(y, MA, v))
# op1 = LinearOperator(Float64, nu, nu, true, true, (y, v) -> ldiv!(y, MA, v))
# # op1 = LinearOperator(Float64, nu, nu, true, true, (y, v) -> ldiv_ilu!(y, L, U, v, z))
# # op1 = LinearOperator(Float64, nu, nu, true, true, (y, v) -> ldiv_chol!(y, MA, v))
# # op1 = LinearOperator(Float64, nu, nu, true, true, (y, v) -> ldiv_ic0!(MA, v, y, z))
# # op1 = LinearOperator(Float64, nu, nu, true, true, (y, v) -> ldiv_ilu0!(y, MA, v, z))
# # op2 = LinearOperator(Float64, np-1, np-1, true, true, (y, v) -> mul!(y, MS, v))
# # op2 = LinearOperator(Float64, np-1, np-1, true, true, (y, v) -> ldiv_chol!(y, MS, v))
# solver = CgSolver(np-1, np-1, VT)
# solver.x .= 0
# M_p = on_architecture(arch, M_p)
# function mass_solve!(y, v)
#     Krylov.solve!(solver, M_p, v, M=MS, ldiv=false, itmax=4)
#     # println("niter=", solver.stats.niter)
#     y .= solver.x
#     return y
# end
# op2 = LinearOperator(Float64, np-1, np-1, true, true, (y, v) -> mass_solve!(y, v))
# P_inversion = BlockDiagonalOperator(op1, op2)
# ldiv_P_inversion = false

# P_inversion = Diagonal(on_architecture(arch, 1/h^2*ones(N)))
# ldiv_P_inversion = false

# # compute LU factorization
# @time "lu(LHS)" LHS_inversion_LU = lu(LHS_inversion)

# inversion RHS
RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

# put on GPU, if needed
LHS_inversion = on_architecture(arch, LHS_inversion)
RHS_inversion = on_architecture(arch, RHS_inversion)
if typeof(arch) == GPU
    CUDA.memory_status()
    println()
end

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
                  atol=atol, rtol=rtol, verbose=1, itmax=itmax, restart=true, history=true)
    @printf("inversion GMRES solve: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
    # solver.x = P_inversion \ RHS
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

# background state \partial_z b = N^2
N² = 1.

# load b from file
statefile = @sprintf("../sims/sim040/data/state001.h5")
ux, uy, uz, p, b, t = load_state(statefile)
ux = interpolate_everywhere(0, Ux)
uy = interpolate_everywhere(0, Uy)
uz = interpolate_everywhere(0, Uz)
p  = interpolate_everywhere(0, P)
b  = FEFunction(B, b)

# true sol
sol_fname = @sprintf("%s/data/sol_2D_%e_%e_%e_%e.jld2", out_folder, hres, ε², γ, f₀)
if isfile(sol_fname)
    file = jldopen(@sprintf("%s/data/sol_2D_%e_%e_%e_%e.jld2", out_folder, hres, ε², γ, f₀), "r")
    ux_true = file["ux"] 
    uy_true = file["uy"] 
    uz_true = file["uz"] 
    p_true = file["p"] 
    close(file)
else
    b_arch = on_architecture(arch, b.free_values)
    RHS = [zeros(nx); zeros(ny); RHS_inversion*b_arch; zeros(np-1)]
    sol = LHS_inversion \ RHS
    sol = sol[inv_perm_inversion]
    ux_true = sol[1:nx]
    uy_true = sol[nx+1:nx+ny]
    uz_true = sol[nx+ny+1:nx+ny+nz]
    p_true = sol[nx+ny+nz+1:end]
    jldsave(@sprintf("%s/data/sol_2D_%e_%e_%e_%e.jld2", out_folder, hres, ε², γ, f₀); ux=ux_true, uy=uy_true, uz=uz_true, p=p_true)
end
ux_true = FEFunction(Ux, ux_true)
uy_true = FEFunction(Uy, uy_true)
uz_true = FEFunction(Uz, uz_true)
p_true  = FEFunction(P, p_true)

# invert
solver_inversion = invert!(arch, solver_inversion, b)
ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

# plot sol
# plots_cache = sim_plots(dim, ux, uy, uz, b, N², H, 0, 0, out_folder)
# plots_cache = sim_plots(dim, ux_true, uy_true, uz_true, b, N², H, 0, 0, out_folder)

# error
eu_H1 = sqrt(sum( ∫( (ux - ux_true)*(ux - ux_true) + (uy - uy_true)*(uy - uy_true) + (uz - uz_true)*(uz - uz_true) + 
                      ∂x(ux - ux_true)*∂x(ux - ux_true) + ∂y(ux - ux_true)*∂y(ux - ux_true) + ∂z(ux - ux_true)*∂z(ux - ux_true) +
                      ∂x(uy - uy_true)*∂x(uy - uy_true) + ∂y(uy - uy_true)*∂y(uy - uy_true) + ∂z(uy - uy_true)*∂z(uy - uy_true) +
                      ∂x(uz - uz_true)*∂x(uz - uz_true) + ∂y(uz - uz_true)*∂y(uz - uz_true) + ∂z(uz - uz_true)*∂z(uz - uz_true) 
                    )*dΩ ))
ep_L2 = sqrt(sum( ∫( (p - p_true)*(p - p_true) )*dΩ ))
@printf("error = %e\n", eu_H1 + ep_L2)

# plot gmres error vs iterations
fig, ax = subplots(1)
# solver_inversion_f = load("$out_folder/data/solver_inversion_f.jld2", "solver_inversion")
# solver_inversion_f_init = load("$out_folder/data/solver_inversion_f_init.jld2", "solver_inversion")
# solver_inversion = load("$out_folder/data/solver_inversion.jld2", "solver_inversion")
# ax.plot(solver_inversion_f.stats.residuals, "-", label=L"$f = 1$")
# ax.plot(solver_inversion_f_init.stats.residuals, "-", label=L"$f = 1$, true init")
# ax.plot(solver_inversion.stats.residuals, "-", label=L"$f = 0$")
ax.plot(solver_inversion.stats.residuals, "-")
# ax.legend()
ax.set_yscale("log")
ax.set_xlabel("Iteration")
ax.set_ylabel("Residual")
savefig("$out_folder/images/gmres_convergence.png")
println("$out_folder/images/gmres_convergence.png")
plt.close()

println("Done.")