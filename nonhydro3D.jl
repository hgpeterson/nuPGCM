using NonhydroPG
using Printf
using Gridap
using GridapGmsh
using Gmsh: gmsh
using IncompleteLU
using Krylov, KrylovPreconditioners
using LinearOperators
using CUDA, CUDA.CUSPARSE, CUDA.CUSOLVER
using SparseArrays
using LinearAlgebra

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
arch = CPU()
# arch = GPU()

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
model = GmshDiscreteModel("bowl3D_0.05.msh")

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
κ(x) = 1e-2 + exp(-(x[2] + H(x))/0.1)

# params
ε² = 1
γ = 1
f(x) = 1
μϱ = 1e-4
Δt = 1e-1
println("\n---")
println("Parameters:\n")
@printf("ε² = %.1e (δ = %.1e, %.1e ≤ h ≤ %.1e)\n", ε², √2ε², hmin, hmax)
@printf(" γ = %.1e\n", γ)
@printf("μϱ = %.1e\n", μϱ)
@printf("Δt = %.1e\n", Δt)
println("---\n")

# inversion LHS
function assemble_LHS_inversion(arch::AbstractArchitecture)
    a_inversion((ux, uy, uz, p), (vx, vy, vz, q)) = 
        ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
           γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
         γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                                                    ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
    @time "assemble LHS_inversion" LHS_inversion = assemble_matrix(a_inversion, X, Y)
    return on_architecture(arch, FT.(LHS_inversion))
end

LHS_inversion = assemble_LHS_inversion(arch)
write_sparse_matrix("out/LHS_inversion.h5", on_architecture(CPU(), LHS_inversion))
# LHS_inversion = on_architecture(arch, read_sparse_matrix("out/LHS_inversion.h5"))
println("eltype(LHS_inversion): ", eltype(LHS_inversion))

# preconditioners for inversion LHS
function compute_P_inversion(::CPU, LHS_inversion)
    @time "LHS_inversion_ilu" P_inversion = ilu(LHS_inversion, τ=1e-6)
end
function compute_P_inversion(::GPU, LHS_inversion)
    return I
    # perm_inversion = zfd(LHS_inversion)
    # perm_inversion .+= 1
    # invperm_inversion = invperm(perm_inversion)
    # LHS_inversion = LHS_inversion[:, perm_inversion]
    # LHS_inversion_gpu = CuSparseMatrixCSR(LHS_inversion)
    # @time "P_inversion" P_inversion = ilu02(LHS_inversion_gpu)

    # # additional vector required for solving triangular systems
    # T = eltype(LHS_inversion_gpu)
    # N = size(LHS_inversion_gpu, 1)
    # temp = CUDA.zeros(T, N)

    # # solve Py = x
    # function ldiv_ilu0!(P::CuSparseMatrixCSR, x, y, temp)
    #     ldiv!(temp, UnitLowerTriangular(P), x)  # forward substitution with L
    #     ldiv!(y, UpperTriangular(P), temp)      # backward substitution with U
    #     return y
    # end

    # # Operator that model P⁻¹
    # symmetric = hermitian = false
    # P_inversion_op = LinearOperator(T, N, N, symmetric, hermitian, (y, x) -> ldiv_ilu0!(P_inversion, x, y, temp))
    # return P_inversion_op
end

P_inversion = compute_P_inversion(arch, LHS_inversion)

# Krylov solver for inversion
# solver_inversion = GmresSolver(N, N, 20, VT)
solver_inversion = BicgstabSolver(N, N, VT)

# inversion functions
function invert!(arch::AbstractArchitecture, solver_inversion, b)
    l_inversion((vx, vy, vz, q)) = ∫( b*vz )dΩ
    RHS_inversion = on_architecture(arch, FT.(assemble_vector(l_inversion, Y)))
    @time "invert!" Krylov.solve!(solver_inversion, LHS_inversion, RHS_inversion, solver_inversion.x, M=P_inversion, ldiv=true)
    return solver_inversion
end
function update_u_p!(ux, uy, uz, p, solver_inversion)
    ux.free_values .= on_architecture(CPU(), solver_inversion.x[1:nx])
    uy.free_values .= on_architecture(CPU(), solver_inversion.x[nx+1:nx+ny])
    uz.free_values .= on_architecture(CPU(), solver_inversion.x[nx+ny+1:nx+ny+nz])
    p = FEFunction(P, on_architecture(CPU(), solver_inversion.x[nx+ny+nz+1:end]))
    return ux, uy, uz, p
end

# initial condition
b0(x) = x[3]
b = interpolate_everywhere(b0, B)
solver_inversion = invert!(arch, solver_inversion, b)
if any(isnan.(solver_inversion.x))
    error("Inversion failed 😿")
end
ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)
i_save = 0
save(ux, uy, uz, p, b, i_save)
i_save += 1

# evolution LHS
function assemble_LHS_evolution(arch::AbstractArchitecture)
    # b^n+1 - Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n+1)) = b^n - Δt*u^n⋅∇b^n + Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n))
    a_evolution(b, d) = ∫( b*d + Δt/2*ε²/μϱ*∂z(b)*∂z(d)*κ )dΩ
    @time "assemble LHS_evolution" LHS_evolution = assemble_matrix(a_evolution, B, D)
    return on_architecture(arch, FT.(LHS_evolution))
end

LHS_evolution = assemble_LHS_evolution(arch)
write_sparse_matrix("out/LHS_evolution.h5", on_architecture(CPU(), LHS_evolution))
# LHS_evolution = on_architecture(arch, read_sparse_matrix("out/LHS_evolution.h5"))
println("eltype(LHS_evolution): ", eltype(LHS_evolution))

# preconditioners for evolution LHS
function compute_P_evolution(::CPU)
    @time "LHS_evolution_ilu" P_evolution = ilu(LHS_evolution, τ=1e-8)
    return P_evolution
end
function compute_P_evolution(::GPU)
    return I
end

P_evolution = compute_P_evolution(arch)

if typeof(arch) == GPU
    CUDA.memory_status()
end

# Krylov solver for evolution
# solver_evolution = GmresSolver(nb, nb, 20, VT)
solver_evolution = BicgstabSolver(nb, nb, VT)

# evolution functions
function evolve!(arch::AbstractArchitecture, solver_evolution, ux, uy, uz, b)
    l_evolution(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - Δt/2*ε²/μϱ*∂z(b)*∂z(d)*κ )dΩ
    RHS_evolution = on_architecture(arch, FT.(assemble_vector(l_evolution, D)))
    @time "evolve!" Krylov.solve!(solver_evolution, LHS_evolution, RHS_evolution, solver_evolution.x, M=P_evolution, ldiv=true)
    return solver_evolution
end
function update_b!(b, solver_evolution)
    b.free_values .= on_architecture(CPU(), solver_evolution.x)
    return b
end

# solve function
function solve!(arch::AbstractArchitecture, ux, uy, uz, p, b, solver_inversion, solver_evolution, i_save, N)
    for i ∈ 1:N
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
            println("\n---")
            @printf("t = %.1f (i = %d, Δt = %.1f)\n\n", i*Δt, i, Δt)
            @printf("|u|ₘₐₓ = %.1e, %.1f ≤ b ≤ %.1f\n", max(maximum(abs.(ux.free_values)), maximum(abs.(uy.free_values)), maximum(abs.(uz.free_values))), minimum(b.free_values), maximum([b.free_values; 0]))
            @printf("CFL ≈ %.5f\n", min(hmin/maximum(abs.(ux.free_values)), hmin/maximum(abs.(uy.free_values)), hmin/maximum(abs.(uz.free_values))))
            println("---\n")
        end
        if mod(i, 1) == 0
            save(ux, uy, uz, p, b, i_save)
            i_save += 1
        end
    end
    return ux, uy, uz, p, b
end

# run
ux, uy, uz, p, b = solve!(arch, ux, uy, uz, p, b, solver_inversion, solver_evolution, i_save, 50)