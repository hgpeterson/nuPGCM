using NonhydroPG
using Gridap, GridapGmsh
using IncompleteLU, Krylov, LinearOperators
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays, LinearAlgebra
using Printf
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

# define CPU and GPU architectures
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture end
struct GPU <: AbstractArchitecture end

# convert types from one architecture to another
on_architecture(::CPU, a::Array) = a
on_architecture(::GPU, a::Array) = CuArray(a)

on_architecture(::CPU, a::CuArray) = Array(a)
on_architecture(::GPU, a::CuArray) = a

on_architecture(::CPU, a::SparseMatrixCSC) = a
on_architecture(::GPU, a::SparseMatrixCSC) = CuSparseMatrixCSR(a)

on_architecture(::CPU, a::CuSparseMatrixCSR) = SparseMatrixCSC(a)
on_architecture(::GPU, a::CuSparseMatrixCSR) = a

# choose architecture
# arch = CPU()
arch = GPU()

# Float type on CPU and GPU
# FT = typeof(arch) == CPU ? Float64 : Float32
FT = Float64

# Vector type on CPU and GPU
VT = typeof(arch) == CPU ? Vector{FT} : CuVector{FT}

# save to vtu
function save(ux, uy, uz, p, b, i)
    fname = @sprintf("out/nonhydro3D%03d.vtu", i)
    writevtk(Ω, fname, cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])
    println(fname)
end

# model
hres = 0.02
model = GmshDiscreteModel(@sprintf("bowl3D_%0.2f.msh", hres))

# mesh res
pts, conns = get_p_t(model)
h1 = [norm(pts[conns[i, 1], :] - pts[conns[i, 2], :]) for i ∈ axes(conns, 1)]
h2 = [norm(pts[conns[i, 2], :] - pts[conns[i, 3], :]) for i ∈ axes(conns, 1)]
h3 = [norm(pts[conns[i, 3], :] - pts[conns[i, 4], :]) for i ∈ axes(conns, 1)]
h4 = [norm(pts[conns[i, 4], :] - pts[conns[i, 1], :]) for i ∈ axes(conns, 1)]
hmin = minimum([h1; h2; h3; h4])
hmax = maximum([h1; h2; h3; h4])

# reference FE 
order = 2
reffe_ux = ReferenceFE(lagrangian, Float64, order;   space=:P)
reffe_uy = ReferenceFE(lagrangian, Float64, order;   space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, order;   space=:P)
reffe_p  = ReferenceFE(lagrangian, Float64, order-1; space=:P)
reffe_b  = ReferenceFE(lagrangian, Float64, order;   space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot"])
Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
D  = TestFESpace(model, reffe_b,  conformity=:H1, dirichlet_tags=["sfc"])
Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0])
Uy = TrialFESpace(Vy, [0])
Uz = TrialFESpace(Vz, [0, 0])
P  = TrialFESpace(Q)
B  = TrialFESpace(D, [0])
X  = MultiFieldFESpace([Ux, Uy, Uz, P])
nx = Ux.space.nfree
ny = Uy.space.nfree
nz = Uz.space.nfree
nu = nx + ny + nz
np = P.space.space.nfree
nb = B.space.nfree
N = nu + np - 1
@printf("\nN = %d (%d + %d) ∼ 10^%d DOF\n", N, nu, np-1, floor(log10(N)))

# initialize vectors
ux = interpolate_everywhere(0, Ux)
uy = interpolate_everywhere(0, Uy)
uz = interpolate_everywhere(0, Uz)
p  = interpolate_everywhere(0, P)

# triangulation and integration measure
degree = order^2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# gradients 
∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

# depth
H(x) = sqrt(2 - x[1]^2 - x[2]^2) - 1

# forcing
ν(x) = 1
κ(x) = 1e-2 + exp(-(x[3] + H(x))/0.1)

# params
ε² = 1e-2
γ = 1/4
f₀ = 1
β = 0
f(x) = f₀ + β*x[2]
μϱ = 1e0
Δt = 1e-4*μϱ/ε²
α = Δt/2*ε²/μϱ # for timestep
println("\n---")
println("Parameters:\n")
@printf("ε² = %.1e (δ = %.1e, %.1e ≤ h ≤ %.1e)\n", ε², √(2ε²), hmin, hmax)
@printf(" γ = %.1e\n", γ)
@printf("μϱ = %.1e\n", μϱ)
@printf("Δt = %.1e\n", Δt)
println("---\n")

# filenames for LHS matrices
LHS_inversion_fname = @sprintf("out/LHS_inversion_%e_%e_%e_%e_%e.h5", hres, ε², γ, f₀, β)
LHS_evolution_fname = @sprintf("out/LHS_evolution_%e_%e.h5", hres, α)
# println(LHS_inversion_fname)
# println(LHS_evolution_fname)

# inversion LHS
function assemble_LHS_inversion()
    a_inversion((ux, uy, uz, p), (vx, vy, vz, q)) = 
        ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
           γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
         γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                                                      ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
    @time "assemble LHS_inversion" LHS_inversion = assemble_matrix(a_inversion, X, Y)
    write_sparse_matrix(LHS_inversion_fname, LHS_inversion)
    return LHS_inversion
end

# LHS_inversion = assemble_LHS_inversion()
LHS_inversion = read_sparse_matrix(LHS_inversion_fname)

# Cuthill-McKee DOF reordering
a_m(u, v) = ∫( u*v )dΩ
@time "RCM perm" begin 
M_ux = assemble_matrix(a_m, Ux, Vx)
M_uy = assemble_matrix(a_m, Uy, Vy)
M_uz = assemble_matrix(a_m, Uz, Vz)
M_p  = assemble_matrix(a_m, P, Q)
perm_ux = CUSOLVER.symrcm(M_ux) .+ 1
perm_uy = CUSOLVER.symrcm(M_uy) .+ 1
perm_uz = CUSOLVER.symrcm(M_uz) .+ 1
perm_p  = CUSOLVER.symrcm(M_p)  .+ 1
perm_inversion = [perm_ux; 
                  perm_uy .+ nx; 
                  perm_uz .+ nx .+ ny; 
                  perm_p  .+ nx .+ ny .+ nz]
end
@time "inv_perm" inv_perm_inversion = invperm(perm_inversion)
# plot_sparsity_pattern(LHS_inversion, fname="images/LHS_inversion.png")
@time "LHS_inversion_perm" LHS_inversion = LHS_inversion[perm_inversion, perm_inversion]
# plot_sparsity_pattern(LHS_inversion, fname="images/LHS_inversion_symrcm.png")

# put on GPU, if needed
LHS_inversion = on_architecture(arch, FT.(LHS_inversion))
if typeof(arch) == GPU
    println()
    CUDA.memory_status()
    println()
end

# preconditioners for inversion LHS
function compute_P_inversion(::CPU, LHS_inversion)
    @time "LHS_inversion_ilu" P_inversion = ilu(LHS_inversion, τ=1e-6)
end
function compute_P_inversion(::GPU, LHS_inversion)
    return I

    # LHS_inversion_cpu = on_architecture(CPU(), LHS_inversion)
    # perm_inversion = zfd(LHS_inversion_cpu)
    # perm_inversion .+= 1
    # invperm_inversion = invperm(perm_inversion)
    # LHS_inversion = on_architecture(GPU(), LHS_inversion_cpu[:, perm_inversion])

    # @time "P_inversion" P_inversion = ilu02(LHS_inversion)

    # # additional vector required for solving triangular systems
    # temp = CUDA.zeros(FT, N)

    # # solve Py = x
    # function ldiv_ilu0!(P::CuSparseMatrixCSR, x, y, temp)
    #     ldiv!(temp, UnitLowerTriangular(P), x)  # forward substitution with L
    #     ldiv!(y, UpperTriangular(P), temp)      # backward substitution with U
    #     return y
    # end

    # # Operator that models P⁻¹
    # P_inversion_op = LinearOperator(FT, N, N, false, false, (y, x) -> ldiv_ilu0!(P_inversion, x, y, temp))

    # @time "P_inversion" P_inversion = ilu(on_architecture(CPU(), LHS_inversion), τ=0.1) 
    # L = on_architecture(GPU(), P_inversion.L)
    # U = on_architecture(GPU(), SparseMatrixCSC(P_inversion.U'))
    # temp = CUDA.zeros(FT, N)
    # function ldiv_ilu!(L, U, x, y, temp)
    #     ldiv!(temp, L, y)  # forward substitution with L
    #     ldiv!(x, U, temp)  # backward substitution with U
    #     return x
    # end
    # P_inversion_op = LinearOperator(FT, N, N, false, false, (x, y) -> ldiv_ilu!(L, U, x, y, temp))
    # return P_inversion_op
end

P_inversion = compute_P_inversion(arch, LHS_inversion)

# Krylov solver for inversion
# solver_inversion = GmresSolver(N, N, 20, VT) # can't use on GPU, too much memory
solver_inversion = BicgstabSolver(N, N, VT)
solver_inversion.x .= on_architecture(arch, zeros(FT, N))

# inversion functions
function invert!(arch::AbstractArchitecture, solver_inversion, b)
    l_inversion((vx, vy, vz, q)) = ∫( b*vz )dΩ
    RHS_inversion = on_architecture(arch, 
                                    FT.(assemble_vector(l_inversion, Y)[perm_inversion])
                                   )
    @time "invert!" Krylov.solve!(solver_inversion, LHS_inversion, RHS_inversion, solver_inversion.x, M=P_inversion, ldiv=true)
    return solver_inversion
end
function update_u_p!(ux, uy, uz, p, solver_inversion)
    sol = on_architecture(CPU(), solver_inversion.x[inv_perm_inversion])
    ux.free_values .= sol[1:nx]
    uy.free_values .= sol[nx+1:nx+ny]
    uz.free_values .= sol[nx+ny+1:nx+ny+nz]
    p = FEFunction(P, sol[nx+ny+nz+1:end])
    return ux, uy, uz, p
end

# initial condition
b0(x) = x[3]
# b0(x) = x[3] + 0.1*exp(-(x[3] + H(x))/0.1)
b = interpolate_everywhere(b0, B)
solver_inversion = invert!(arch, solver_inversion, b)
ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)
i_save = 0
plot_profiles(ux, uy, uz, b, 0.5, 0.0, H; t=0, fname=@sprintf("images/profiles%03d.png", i_save))
save(ux, uy, uz, p, b, i_save)
i_save += 1

# evolution LHS
function assemble_LHS_evolution()
    # b^n+1 - Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n+1)) = b^n - Δt*u^n⋅∇b^n + Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n))
    a_evolution(b, d) = ∫( b*d + α*∂z(b)*∂z(d)*κ )dΩ
    @time "assemble LHS_evolution" LHS_evolution = assemble_matrix(a_evolution, B, D)
    write_sparse_matrix(LHS_evolution_fname, LHS_evolution)
    return LHS_evolution
end

LHS_evolution = assemble_LHS_evolution()
# LHS_evolution = read_sparse_matrix(LHS_evolution_fname)

# Cuthill-McKee DOF reordering
@time "RCM perm" begin
M_b = assemble_matrix(a_m, B, D)
perm_evolution = CUSOLVER.symrcm(M_b) .+ 1
end
@time "inv_perm" inv_perm_evolution = invperm(perm_evolution)
# plot_sparsity_pattern(LHS_evolution, fname="images/LHS_evolution.png")
@time "LHS_evolution_perm" LHS_evolution = LHS_evolution[perm_evolution, perm_evolution]
# plot_sparsity_pattern(LHS_evolution, fname="images/LHS_evolution_symrcm.png")

# put on GPU, if needed
LHS_evolution = on_architecture(arch, FT.(LHS_evolution))
if typeof(arch) == GPU
    println()
    CUDA.memory_status()
    println()
end

# preconditioners for evolution LHS
function compute_P_evolution(::CPU)
    @time "LHS_evolution_ilu" P_evolution = ilu(LHS_evolution, τ=1e-10)
    # @time "LHS_evolution_ilu" P_evolution = lu(LHS_evolution)
    return P_evolution
end
function compute_P_evolution(::GPU)
    return I
end

P_evolution = compute_P_evolution(arch)

# Krylov solver for evolution
# solver_evolution = GmresSolver(nb, nb, 20, VT)
# solver_evolution = BicgstabSolver(nb, nb, VT)
solver_evolution = CgSolver(nb, nb, VT)
solver_evolution.x .= on_architecture(arch, copy(b.free_values))

# evolution functions
function evolve!(arch::AbstractArchitecture, solver_evolution, ux, uy, uz, b)
    l_evolution(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - α*∂z(b)*∂z(d)*κ )dΩ
    RHS_evolution = on_architecture(arch, 
                                    FT.(assemble_vector(l_evolution, D)[perm_evolution])
                                   )
    @time "evolve!" Krylov.solve!(solver_evolution, LHS_evolution, RHS_evolution, solver_evolution.x, M=P_evolution, ldiv=true)
    return solver_evolution
end
function update_b!(b, solver_evolution)
    b.free_values .= on_architecture(CPU(), solver_evolution.x[inv_perm_evolution])
    return b
end

# solve function
function solve!(arch::AbstractArchitecture, ux, uy, uz, p, b, solver_inversion, solver_evolution, i_save, n_steps)
    t0 = time()
    for i ∈ 1:n_steps
        # evolve
        solver_evolution = evolve!(arch, solver_evolution, ux, uy, uz, b)
        b = update_b!(b, solver_evolution)

        # invert
        solver_inversion = invert!(arch, solver_inversion, b)
        ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

        if any(isnan.(solver_inversion.x)) || any(isnan.(solver_evolution.x))
            error("Solution diverged 🤯")
        end

        # info/save
        if mod(i, 10) == 0
            t1 = time()
            println("\n---")
            @printf("t = %.1f (i = %d, Δt = %f)\n\n", i*Δt, i, Δt)
            @printf("time elapsed: %02d:%02d:%02d\n", hrs_mins_secs(t1-t0)...)
            @printf("estimated time remaining: %02d:%02d:%02d\n", hrs_mins_secs((t1-t0)*(n_steps-i)/i)...)
            @printf("|u|ₘₐₓ = %.1e, %.1e ≤ b ≤ %.1e\n", max(maximum(abs.(ux.free_values)), maximum(abs.(uy.free_values)), maximum(abs.(uz.free_values))), minimum(b.free_values), maximum([b.free_values; 0]))
            @printf("CFL ≈ %.5f\n", min(hmin/maximum(abs.(ux.free_values)), hmin/maximum(abs.(uy.free_values)), hmin/maximum(abs.(uz.free_values))))
            println("---\n")

            plot_profiles(ux, uy, uz, b, 0.5, 0.0, H; t=i*Δt, fname=@sprintf("images/profiles%03d.png", i_save))
            save(ux, uy, uz, p, b, i_save)
            i_save += 1
        end
    end
    return ux, uy, uz, p, b
end

function hrs_mins_secs(seconds)
    return seconds ÷ 3600, (seconds % 3600) ÷ 60, seconds % 60
end

# run
ux, uy, uz, p, b = solve!(arch, ux, uy, uz, p, b, solver_inversion, solver_evolution, i_save, 500)