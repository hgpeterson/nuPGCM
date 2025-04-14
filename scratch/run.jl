using nuPGCM
using Gridap
using GridapGmsh
using Krylov
using CUDA
using LinearAlgebra
using JLD2
using Printf
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

set_out_dir!(".")

ENV["JULIA_DEBUG"] = nuPGCM
# ENV["JULIA_DEBUG"] = nothing

# choose dimensions
dim = TwoD()
# dim = ThreeD()

# choose architecture
# arch = CPU()
arch = GPU()

# model
h = 0.01
model = GmshDiscreteModel(@sprintf("../meshes/bowl%s_%0.2f.msh", dim, h))

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
@info @sprintf("N = %d (%d + %d) ∼ 10^%d DOF", N, nu, np-1, floor(log10(N)))

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
β = 0
f(x) = f₀ + β*x[2]
μϱ = 1e0
Δt = 1e-4*μϱ/ε²
T = 5e-2*μϱ/ε²
α = Δt/2*ε²/μϱ # for timestep
println("\n---")
println("Parameters:\n")
@printf("ε² = %.1e (δ = %.1e, h ≈ %.1e)\n", ε², √(2ε²), h)
@printf("f₀ = %.1e\n", f₀)
@printf(" β = %.1e\n", β)
@printf(" γ = %.1e\n", γ)
@printf("μϱ = %.1e\n", μϱ)
@printf("Δt = %.1e\n", Δt)
@printf(" T = %.1e\n", T)
println("---\n")

# inversion LHS
LHS_inversion_fname = @sprintf("../matrices/LHS_inversion_%s_%e_%e_%e_%e_%e.h5", dim, h, ε², γ, f₀, β)
# if isfile(LHS_inversion_fname)
    # LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
# else
    LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
# end

# inversion RHS
RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

# preconditioner
if typeof(dim) == TwoD && typeof(arch) == CPU
    @time "lu(LHS_inversion)" P_inversion = lu(LHS_inversion)
else
    P_inversion = Diagonal(on_architecture(arch, 1/h^dim.n*ones(N)))
    # P_inversion = I
end

# move to arch
LHS_inversion = on_architecture(arch, LHS_inversion)
RHS_inversion = on_architecture(arch, RHS_inversion)
typeof(arch) == GPU() && CUDA.memory_status()

# setup inversion toolkit
inversion_toolkit = InversionToolkit(LHS_inversion, P_inversion, RHS_inversion)

function update_u_p!(ux, uy, uz, p, solver)
    sol = on_architecture(CPU(), solver.x[inv_perm_inversion])
    ux.free_values .= sol[1:nx]
    uy.free_values .= sol[nx+1:nx+ny]
    uz.free_values .= sol[nx+ny+1:nx+ny+nz]
    p = FEFunction(P, sol[nx+ny+nz+1:end])
    return ux, uy, uz, p
end

# background state ∂z(b) = N²
N² = 1.

# initial condition: b = N²z, t = 0
i_save = 0
b = interpolate_everywhere(0, B)
t = 0.
ux = interpolate_everywhere(0, Ux)
uy = interpolate_everywhere(0, Uy)
uz = interpolate_everywhere(0, Uz)
p  = interpolate_everywhere(0, P) 
# save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_dir, i_save))

# # initial condition: b = N²z + exponential
# i_save = 0
# b = interpolate_everywhere(x -> 0.1*exp(-(x[3] + H(x))/0.1), B)
# t = 0.
# ux = interpolate_everywhere(0, Ux)
# uy = interpolate_everywhere(0, Uy)
# uz = interpolate_everywhere(0, Uz)
# p  = interpolate_everywhere(0, P) 

# # initial condition: load from file
# i_save = 1
# statefile = @sprintf("%s/data/state%03d.h5", out_dir, i_save)
# ux, uy, uz, p, b, t = load_state(statefile)
# solver_inversion.x .= on_architecture(arch, [ux; uy; uz; p][perm_inversion])
# ux = FEFunction(Ux, ux)
# uy = FEFunction(Uy, uy)
# uz = FEFunction(Uz, uz)
# p  = FEFunction(P, p)
# b  = FEFunction(B, b)

# # initial condition: load from diffusion
# diff_dir = "../sims/sim048"
# i_save = 50
# file = jldopen(@sprintf("%s/data/b%03d.jld2", diff_dir, i_save), "r")
# b = FEFunction(B, file["b"])
# t = file["t"]
# close(file)
# ux = interpolate_everywhere(0, Ux)
# uy = interpolate_everywhere(0, Uy)
# uz = interpolate_everywhere(0, Uz)
# p  = interpolate_everywhere(0, P) 

# # invert initial condition
# invert!(inversion_toolkit, b)
# ux, uy, uz, p = update_u_p!(ux, uy, uz, p, inversion_toolkit.solver)
# save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_dir, i_save))

# # plot
# plots_cache = sim_plots(dim, ux, uy, uz, b, N², H, t, i_save)
# i_save += 1

# evolution LHSs
LHS_diff_fname = @sprintf("../matrices/LHS_diff_%s_%e_%e_%e.h5", dim, h, α, γ)
LHS_adv_fname = @sprintf("../matrices/LHS_adv_%s_%e.h5", dim, h)
if isfile(LHS_adv_fname) && isfile(LHS_diff_fname)
    LHS_adv,  perm_b, inv_perm_b = read_sparse_matrix(LHS_adv_fname)
    LHS_diff, perm_b, inv_perm_b = read_sparse_matrix(LHS_diff_fname)
else
    LHS_adv, LHS_diff, perm_b, inv_perm_b = assemble_LHS_adv_diff(arch, α, γ, κ, B, D, dΩ; fname_adv=LHS_adv_fname, fname_diff=LHS_diff_fname)
end

# diffusion RHS matrix and vector
RHS_diff, rhs_diff = assemble_RHS_diff(perm_b, α, γ, κ, N², B, D, dΩ)

# preconditioners
if typeof(dim) == TwoD && typeof(arch) == CPU
    @time "lu(LHS_diff)" P_diff = lu(LHS_diff)
    @time "lu(LHS_adv)"  P_adv  = lu(LHS_adv)
else
    P_diff = Diagonal(on_architecture(arch, Vector(1 ./ diag(LHS_diff))))
    P_adv  = Diagonal(on_architecture(arch, Vector(1 ./ diag(LHS_adv))))
end

# move to arch
LHS_adv = on_architecture(arch, LHS_adv)
LHS_diff = on_architecture(arch, LHS_diff)
RHS_diff = on_architecture(arch, RHS_diff)
rhs_diff = on_architecture(arch, rhs_diff)
typeof(arch) == GPU() && CUDA.memory_status()

# Krylov solver for evolution
VT = typeof(arch) == CPU ? Vector{Float64} : CuVector{Float64}
tol = 1e-6
itmax = 0
solver_evolution = CgSolver(nb, nb, VT)
solver_evolution.x .= on_architecture(arch, copy(b.free_values)[perm_b])

# evolution functions
b_half = interpolate_everywhere(0, B)
function evolve_adv!(inversion_toolkit, solver_evolution, ux, uy, uz, p, b)
    # determine architecture
    arch = architecture(solver_evolution.x)

    # half step
    l_half(d) = ∫( b*d - Δt/2*(ux*∂x(b) + uy*∂y(b) + uz*(N² + ∂z(b)))*d )dΩ
    RHS = on_architecture(arch, assemble_vector(l_half, D)[perm_b])
    Krylov.solve!(solver_evolution, LHS_adv, RHS, solver_evolution.x, M=P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)
    # @printf("advection CG solve 1: solved=%s, niter=%d, time=%f\n", solver_evolution.stats.solved, solver_evolution.stats.niter, solver_evolution.stats.timer)

    # u, v, w, p, b at half step
    update_b!(b_half, solver_evolution)
    invert!(inversion_toolkit, b_half)
    ux, uy, uz, p = update_u_p!(ux, uy, uz, p, inversion_toolkit.solver)

    # full step
    l_full(d) = ∫( b*d - Δt*(ux*∂x(b_half) + uy*∂y(b_half) + uz*(N² + ∂z(b_half)))*d )dΩ
    RHS = on_architecture(arch, assemble_vector(l_full, D)[perm_b])
    Krylov.solve!(solver_evolution, LHS_adv, RHS, solver_evolution.x, M=P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)
    # @printf("advection CG solve 2: solved=%s, niter=%d, time=%f\n", solver_evolution.stats.solved, solver_evolution.stats.niter, solver_evolution.stats.timer)

    return inversion_toolkit, solver_evolution
end
function evolve_diff!(solver, b)
    arch = architecture(solver.x)
    b_arch= on_architecture(arch, b.free_values)
    RHS = RHS_diff*b_arch + rhs_diff
    Krylov.solve!(solver, LHS_diff, RHS, solver.x, M=P_diff, atol=tol, rtol=tol, verbose=0, itmax=itmax)
    # @printf("diffusion CG solve: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
    return solver
end
function update_b!(b, solver)
    b.free_values .= on_architecture(CPU(), solver.x[inv_perm_b])
    return b
end

# solve function
function solve!(ux, uy, uz, p, b, t, inversion_toolkit, solver_evolution, i_save, i_step, n_steps)
    t0 = time()
    for i ∈ i_step:n_steps
        flush(stdout)
        flush(stderr)

        # advection step
        evolve_adv!(inversion_toolkit, solver_evolution, ux, uy, uz, p, b)
        update_b!(b, solver_evolution)

        # diffusion step
        evolve_diff!(solver_evolution, b)
        update_b!(b, solver_evolution)

        # invert
        invert!(inversion_toolkit, b)
        ux, uy, uz, p = update_u_p!(ux, uy, uz, p, inversion_toolkit.solver)

        # blow up
        if any(isnan.(inversion_toolkit.solver.x)) || any(isnan.(solver_evolution.x))
            # save and kill
            save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_dir, i_save))
            sim_plots(plots_cache, ux, uy, uz, b, t, i_save)
            error("Solution diverged: NaN(s) found.")
        end

        # time
        t += Δt

        # info
        if mod(i, n_steps ÷ 100) == 0
            ux_max = maximum(abs.(ux.free_values))
            uy_max = maximum(abs.(uy.free_values))
            uz_max = maximum(abs.(uz.free_values))
            t1 = time()
            @info begin
            msg  = @sprintf("t = %f (i = %d/%d, Δt = %f)\n", t, i, n_steps, Δt)
            msg *= @sprintf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t1-t0)...)
            msg *= @sprintf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)*(n_steps-i)/(i-i_step+1))...)
            msg *= @sprintf("|u|ₘₐₓ = %.1e, %.1e ≤ b′ ≤ %.1e\n", max(ux_max, uy_max, uz_max), minimum([b.free_values; 0]), maximum([b.free_values; 0]))
            # msg *= @sprintf("CFL ≈ %f\n", min(h/ux_max, h/uy_max, h/uz_max))
            msg
            end
        end

        # # blow up
        # if max(ux_max, uy_max, uz_max) > 10
        #     # save and kill
        #     save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_dir, i_save))
        #     sim_plots(plots_cache, ux, uy, uz, b, t, i_save)
        #     error("Solution diverged: |u|ₘₐₓ > 10.")
        # end

        # # save/plot
        # if mod(i, n_steps ÷ 50) == 0
        #     save_state(ux, uy, uz, p, b, t; fname=@sprintf("%s/data/state%03d.h5", out_dir, i_save))
        #     sim_plots(plots_cache, ux, uy, uz, b, t, i_save)
        #     i_save += 1
        # end
    end
    return ux, uy, uz, p, b
end

# run
i_step = Int64(round(t/Δt)) + 1
n_steps = Int64(round(T/Δt))
ux, uy, uz, p, b = solve!(ux, uy, uz, p, b, t, inversion_toolkit, solver_evolution, i_save, i_step, n_steps)

println("Done.")