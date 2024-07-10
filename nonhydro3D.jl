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

function save(ux, uy, uz, p, b, i)
    writevtk(Ω, @sprintf("out/nonhydro3D%04d", i), cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])
end

# model
model = GmshDiscreteModel("bowl3D_0.02.msh")

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
println("N = ", nu+np-1, " (", nu, " + ", np-1, ")")

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
ε² = 1e-3
println("δ = ", √(2ε²))
pts, conns = get_p_t(model)
h1 = [norm(pts[conns[i, 1], :] - pts[conns[i, 2], :]) for i ∈ axes(conns, 1)]
h2 = [norm(pts[conns[i, 2], :] - pts[conns[i, 3], :]) for i ∈ axes(conns, 1)]
h3 = [norm(pts[conns[i, 3], :] - pts[conns[i, 4], :]) for i ∈ axes(conns, 1)]
h4 = [norm(pts[conns[i, 4], :] - pts[conns[i, 1], :]) for i ∈ axes(conns, 1)]
hmin = minimum([h1; h2; h3; h4])
hmax = maximum([h1; h2; h3; h4])
println("hmin = ", hmin)
println("hmax = ", hmax)
γ = 1/2
f(x) = 1
μϱ = 1e0
Δt = 1e-1

# LHS matrix
a_inversion((ux, uy, uz, p), (vx, vy, vz, q)) = 
    ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
       γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
     γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
                                                                  ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
@time "assemble LHS_inversion" LHS_inversion = assemble_matrix(a_inversion, X, Y)
@time "LHS_inversion_ilu" LHS_inversion_ilu = ilu(LHS_inversion, τ=1e-6)
# @time "LHS_inversion_ilu" LHS_inversion_ilu = lu(LHS_inversion)
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
# opM = LinearOperator(T, N, N, symmetric, hermitian, (y, x) -> ldiv_ilu0!(P_inversion, x, y, temp))

function invert!(ux, uy, uz, p, b)
    l_inversion((vx, vy, vz, q)) = ∫( b*vz )dΩ
    RHS_inversion = assemble_vector(l_inversion, Y)
    sol0 = [ux.free_values; uy.free_values; uz.free_values; p.free_values]

    @time "invert" sol, stats = gmres(LHS_inversion, RHS_inversion, sol0, M=LHS_inversion_ilu, ldiv=true)

    # RHS_inversion_gpu = CuVector(RHS_inversion)
    # sol0_gpu = CuVector(sol0[perm_inversion])
    # @time "invert" sol_gpu, stats = gmres(LHS_inversion_gpu, RHS_inversion_gpu, sol0_gpu, M=opM, verbose=1)
    # sol = Vector(sol_gpu)[invperm_inversion]

    # RHS_inversion_gpu = CuVector(RHS_inversion)
    # sol0_gpu = CuVector(sol0)
    # @time "invert" sol_gpu, stats = gmres(opLHS_inversion_gpu, RHS_inversion_gpu, sol0_gpu, verbose=1)
    # sol = Vector(sol_gpu)

    ux.free_values .= sol[1:nx]
    uy.free_values .= sol[nx+1:nx+ny]
    uz.free_values .= sol[nx+ny+1:nx+ny+nz]
    p = FEFunction(P, sol[nx+ny+nz+1:end])
    return ux, uy, uz, p
end

# initial cond
b0(x) = x[3]
# b0(x) = x[3] + 0.1*exp(-(x[3] + H(x))/0.1)
b = interpolate_everywhere(b0, B)
ux, uy, uz, p = invert!(ux, uy, uz, p, b)
# save(ux, uy, uz, p, b, -1)
# error()
i_save = 0
save(ux, uy, uz, p, b, i_save)
i_save += 1

# b^n+1 - Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n+1)) = b^n - Δt*u^n⋅∇b^n + Δt/2*ε²/μϱ ∂z(κ(x) ∂z(b^n))
# evolution LHS
a_evolution(b, d) = ∫( b*d + Δt/2*ε²/μϱ*∂z(b)*∂z(d)*κ )dΩ
@time "assemble LHS_evolution" LHS_evolution = assemble_matrix(a_evolution, B, D)
@time "LHS_evolution_ilu" LHS_evolution_ilu = ilu(LHS_evolution, τ=1e-8)
# @time "LHS_evolution_ilu" LHS_evolution_ilu = lu(LHS_evolution)
# using LinearAlgebra
# @assert issymmetric(LHS_evolution)
# @assert isposdef(LHS_evolution)
# LHS_evolution_gpu = CuSparseMatrixCSR(LHS_evolution)

function solve!(ux, uy, uz, p, b, D, hmin, i_save, N)
    for i ∈ 1:N
        # evolution RHS
        l_evolution(d) = ∫( b*d - Δt*ux*∂x(b)*d - Δt*uy*∂y(b)*d - Δt*uz*∂z(b)*d - Δt/2*ε²/μϱ*∂z(b)*∂z(d)*κ )dΩ
        RHS_evolution = assemble_vector(l_evolution, D)
        # RHS_evolution_gpu = CuVector(RHS_evolution)
        @time "update b" sol, stats = gmres(LHS_evolution, RHS_evolution, b.free_values, M=LHS_evolution_ilu, ldiv=true)
        b.free_values .= sol
        # sol0_gpu = CuVector(b.free_values)
        # @time "update b" sol_gpu, stats = gmres(LHS_evolution_gpu, RHS_evolution_gpu, sol0_gpu)
        # b.free_values .= Vector(sol_gpu)
        ux, uy, uz, p = invert!(ux, uy, uz, p, b)
        if mod(i, 10) == 0
            @printf("% 5d  %1.2e\n", i, min(hmin/maximum(abs.(ux.free_values)), hmin/maximum(abs.(uy.free_values)), hmin/maximum(abs.(uz.free_values))))
        end
        if mod(i, 1) == 0
            save(ux, uy, uz, p, b, i_save)
            i_save += 1
        end
    end
    return ux, uy, uz, p, b
end

ux, uy, uz, p, b = solve!(ux, uy, uz, p, b, D, hmin, i_save, 100000)