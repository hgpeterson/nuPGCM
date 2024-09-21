using NonhydroPG
using Gridap, GridapGmsh
using IncompleteLU, Krylov, LinearOperators, CuthillMcKee
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

out_folder = "sim028"

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

# choose architecture
# arch = CPU()
arch = GPU()

# tolerance and max iterations for iterative solvers
tol = 1e-8
@printf("tol = %.1e\n", tol)
itmax = 0
@printf("itmax = %d\n", itmax)

# Vector type 
VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}

# model
hres = 0.01
model = GmshDiscreteModel(@sprintf("meshes/bowl3D_%0.2f.msh", hres))

# full grid
m = Mesh(model)

# surface grid
m_sfc = Mesh(model, "sfc")

# mesh res
h1 = [norm(m.p[m.t[i, 1], :] - m.p[m.t[i, 2], :]) for i ∈ axes(m.t, 1)]
h2 = [norm(m.p[m.t[i, 2], :] - m.p[m.t[i, 3], :]) for i ∈ axes(m.t, 1)]
h3 = [norm(m.p[m.t[i, 3], :] - m.p[m.t[i, 4], :]) for i ∈ axes(m.t, 1)]
h4 = [norm(m.p[m.t[i, 4], :] - m.p[m.t[i, 1], :]) for i ∈ axes(m.t, 1)]
hmin = minimum([h1; h2; h3; h4])
hmax = maximum([h1; h2; h3; h4])

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
κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

# params
ε² = 1e-4
γ = 1/4
f₀ = 1
β = 1
f(x) = f₀ + β*x[2]
μϱ = 1e0
# μϱ = 1e-4
# Δt = 1e-4*μϱ/ε²
Δt = 0.05
T = 5e-2*μϱ/ε²
α = Δt/2*ε²/μϱ # for timestep
println("\n---")
println("Parameters:\n")
@printf("ε² = %.1e (δ = %.1e, %.1e ≤ h ≤ %.1e)\n", ε², √(2ε²), hmin, hmax)
@printf("f₀ = %.1e\n", f₀)
@printf(" β = %.1e\n", β)
@printf(" γ = %.1e\n", γ)
@printf("μϱ = %.1e\n", μϱ)
@printf("Δt = %.1e\n", Δt)
@printf(" T = %.1e\n", T)
println("---\n")

# filenames for LHS matrices
LHS_inversion_fname = @sprintf("matrices/LHS_inversion_%e_%e_%e_%e_%e.h5", hres, ε², γ, f₀, β)
# LHS_evolution_fname = @sprintf("matrices/LHS_evolution_%e_%e.h5", hres, α)
LHS_evolution_fname = @sprintf("matrices/LHS_evolution_%e_%e_%e.h5", hres, α, γ)

# inversion LHS
# if isfile(LHS_inversion_fname)
#     LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
# else
#     LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
# end
LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)

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

# initial condition: b = z, t = 0
i_save = 0
# b = interpolate_everywhere(x->x[3], B)
b = interpolate_everywhere(0, B)
t = 0.
ux = interpolate_everywhere(0, Ux)
uy = interpolate_everywhere(0, Uy)
uz = interpolate_everywhere(0, Uz)
p  = interpolate_everywhere(0, P)
# solver_inversion = invert!(arch, solver_inversion, b)
# ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)
save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_folder, i_save))

# # initial condition: load from file
# i_save = 18
# statefile = @sprintf("%s/data/state%03d.h5", out_folder, i_save)
# ux, uy, uz, p, b, t = load_state(statefile)
# solver_inversion.x .= on_architecture(arch, [ux; uy; uz; p][perm_inversion])
# ux = FEFunction(Ux, ux)
# uy = FEFunction(Uy, uy)
# uz = FEFunction(Uz, uz)
# p  = FEFunction(P, p)
# b  = FEFunction(B, b)

# plot initial condition
plots_cache = sim_plots(ux, uy, uz, b, H, t, i_save, out_folder)
i_save += 1

# if isfile(LHS_evolution_fname)
#     LHS_evolution, perm_evolution, inv_perm_evolution  = read_sparse_matrix(LHS_evolution_fname)
# else
#     LHS_evolution, perm_evolution, inv_perm_evolution = assemble_LHS_evolution(arch, α, γ, κ, B, D, dΩ; fname=LHS_evolution_fname)
# end
LHS_evolution, perm_evolution, inv_perm_evolution = assemble_LHS_evolution(arch, α, γ, κ, B, D, dΩ; fname=LHS_evolution_fname)

# preconditioner
P_evolution = Diagonal(Vector(1 ./ diag(LHS_evolution)))

# put on GPU, if needed
LHS_evolution = on_architecture(arch, LHS_evolution)
P_evolution = Diagonal(on_architecture(arch, diag(P_evolution)))
if typeof(arch) == GPU
    CUDA.memory_status()
    println()
end

# Krylov solver for evolution
solver_evolution = CgSolver(nb, nb, VT)
solver_evolution.x .= on_architecture(arch, copy(b.free_values)[perm_evolution])

# evolution functions
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)
assembler = SparseMatrixAssembler(D, D)
RHS_evolution = zeros(nb)
function evolve!(arch::AbstractArchitecture, solver, ux, uy, uz, b)
    # l(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - α*γ*∂x(b)*∂x(d)*κ - α*γ*∂y(b)*∂y(d)*κ - α*∂z(b)*∂z(d)*κ )dΩ
    # l(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*(1 + ∂z(b))*d - α*γ*∂x(b)*∂x(d)*κ - α*γ*∂y(b)*∂y(d)*κ - α*(1 + ∂z(b))*∂z(d)*κ )dΩ
    # l(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - Δt*uz*d - α*γ*∂x(b)*∂x(d)*κ - α*γ*∂y(b)*∂y(d)*κ - α*∂z(b)*∂z(d)*κ - 2α*∂z(d)*κ )dΩ
    l(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - Δt*uz*d - α*∂z(b)*∂z(d)*κ - 2α*∂z(d)*κ )dΩ
    # l(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - α*∂z(b)*∂z(d)*κ )dΩ
    # l(d) = ∫( b*d - α*∂z(b)*∂z(d)*κ )dΩ
    # @time "build RHS_evolution" RHS = on_architecture(arch, assemble_vector(l, D)[perm_evolution])
    @time "build RHS_evolution" Gridap.FESpaces.assemble_vector!(l, RHS_evolution, assembler, D)
    RHS = on_architecture(arch, RHS_evolution[perm_evolution])
    Krylov.solve!(solver, LHS_evolution, RHS, solver.x, M=P_evolution,
                  atol=tol, rtol=tol, verbose=0, itmax=itmax)
    @printf("evolution CG solve: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
    return solver
end
function update_b!(b, solver)
    b.free_values .= on_architecture(CPU(), solver.x[inv_perm_evolution])
    return b
end

# solve function
function solve!(arch::AbstractArchitecture, ux, uy, uz, p, b, t, solver_inversion, solver_evolution, i_save, i_step, n_steps)
    t0 = time()
    for i ∈ i_step:n_steps
        flush(stdout)
        flush(stderr)

        # evolve
        solver_evolution = evolve!(arch, solver_evolution, ux, uy, uz, b)
        b = update_b!(b, solver_evolution)

        # invert
        solver_inversion = invert!(arch, solver_inversion, b)
        ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

        if any(isnan.(solver_inversion.x)) || any(isnan.(solver_evolution.x))
            error("Solution diverged 🤯")
        end

        # time
        t += Δt

        # info
        t1 = time()
        println("\n---")
        @printf("t = %f (i = %d/%d, Δt = %f)\n\n", t, i, n_steps, Δt)
        @printf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t1-t0)...)
        @printf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)*(n_steps-i)/(i-i_step+1))...)
        @printf("|u|ₘₐₓ = %.1e, %.1e ≤ b ≤ %.1e\n", max(maximum(abs.(ux.free_values)), maximum(abs.(uy.free_values)), maximum(abs.(uz.free_values))), minimum(b.free_values), maximum([b.free_values; 0]))
        @printf("CFL ≈ %f\n", min(hmin/maximum(abs.(ux.free_values)), hmin/maximum(abs.(uy.free_values)), hmin/maximum(abs.(uz.free_values))))
        println("---\n")

        # save/plot
        if mod(i, n_steps ÷ 50) == 0
            save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_folder, i_save))
            sim_plots(plots_cache, ux, uy, uz, b, t, i_save, out_folder)
            i_save += 1
        end
    end
    return ux, uy, uz, p, b
end

# run
i_step = Int64(round(t/Δt)) + 1
n_steps = Int64(round(T/Δt))
ux, uy, uz, p, b = solve!(arch, ux, uy, uz, p, b, t, solver_inversion, solver_evolution, i_save, i_step, n_steps)