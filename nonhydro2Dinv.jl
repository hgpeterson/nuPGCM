using NonhydroPG
using Gridap
using GridapGmsh
using Gmsh: gmsh
using IncompleteLU
using IterativeSolvers
using Krylov
using LinearMaps
using SparseArrays
using Preconditioners
using LinearAlgebra
import LinearAlgebra: ldiv!

# model
model = GmshDiscreteModel("bowl2D.msh")
# writevtk(model, "model")

# reference FE 
order = 2
reffe_ux = ReferenceFE(lagrangian, Float64, order;   space=:P)
reffe_uy = ReferenceFE(lagrangian, Float64, order;   space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, order;   space=:P)
reffe_p  = ReferenceFE(lagrangian, Float64, order-1; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot"])
Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot"])
# Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot", "sfc"])
# Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot", "sfc"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0])
Uy = TrialFESpace(Vy, [0])
# g(x) = sign(x[1])
# Ux = TrialFESpace(Vx, [0, g])
# Uy = TrialFESpace(Vy, [0, 0])
Uz = TrialFESpace(Vz, [0, 0])
P  = TrialFESpace(Q)
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
ux.dirichlet_values .= Ux.dirichlet_values
uy.dirichlet_values .= Uy.dirichlet_values
uz.dirichlet_values .= Uz.dirichlet_values

# triangulation and integration measure
degree = order^2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# gradients 
x = VectorValue(1.0, 0.0)
z = VectorValue(0.0, 1.0)
∂x(u) = x⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
# # b(x) = x[1]
# δ = 0.1
# H(x) = 1 - x[1]^2
# b(x) = x[2] + δ*exp(-(x[2] + H(x))/δ)
b(x) = 0
ν(x) = 1
# ν(x) = 1e-2 + exp(-(x[2] + H(x))/0.1)

# params
ε² = 1e-2
println(sqrt(2ε²))
γ = 1
f = 1

# bilinear and linear form
a((ux, uy, uz, p), (vx, vy, vz, q)) = 
    ∫( γ*ε²*∂x(ux)*∂x(vx)*ν(x) +   ε²*∂z(ux)*∂z(vx)*ν(x) - f*uy*vx + ∂x(p)*vx +
       γ*ε²*∂x(uy)*∂x(vy)*ν(x) +   ε²*∂z(uy)*∂z(vy)*ν(x) + f*ux*vy            +
     γ^2*ε²*∂x(uz)*∂x(vz)*ν(x) + γ*ε²*∂z(uz)*∂z(vz)*ν(x) +           ∂z(p)*vz +
                                                          ∂x(ux)*q + ∂z(uz)*q )dΩ
    # ∫( γ*ε²*∂x(ux)*∂x(vx)*ν(x) +   ε²*∂z(ux)*∂z(vx)*ν(x) + ∂x(p)*vx +
    #    γ*ε²*∂x(uy)*∂x(vy)*ν(x) +   ε²*∂z(uy)*∂z(vy)*ν(x)            +
    #  γ^2*ε²*∂x(uz)*∂x(vz)*ν(x) + γ*ε²*∂z(uz)*∂z(vz)*ν(x) + ∂z(p)*vz +
    #                                                       ∂x(ux)*q + ∂z(uz)*q )dΩ
l((vx, vy, vz, q)) = ∫( b*vz )dΩ

# assemble
@time "op" op = AffineFEOperator(a, l, X, Y)
LHS = get_matrix(op.op)
RHS = get_vector(op.op)

# # # exact sol
# # @time "solve(op)" ux, uy, uz, p = solve(op)
# # writevtk(Ω, "out/nonhydro2D_true", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])

# # exact sol (manual)
# @time "LHS\\RHS" sol = LHS\RHS
# ux.free_values .= sol[1:nx]
# uy.free_values .= sol[nx+1:nx+ny]
# uz.free_values .= sol[nx+ny+1:nx+ny+nz]
# p = FEFunction(P, sol[nx+ny+nz+1:end])
# writevtk(Ω, "out/nonhydro2D_true", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])

# # # iterate on full matrix
# # @time "LHS_ilu" LHS_ilu = ilu(LHS, τ=1e-5)
# # @time "sol" sol, ch = bicgstabl(LHS, RHS, Pl=LHS_ilu, verbose=true, log=true)

# # assemble mass matrix for preconditioning
# a_m(p, q) = ∫( p*q )dΩ
# @time "assemble M" M = assemble_matrix(a_m, P, Q)
# @time "M iLU" M_ilu = ilu(M/ε², τ=1e-10)
# # @time "M LU" M_lu = lu(M/ε²)

# # set up block matrices: LHS = [A B; BT D] where D should be 0; RHS = [F; G]
# A = LHS[1:nu, 1:nu]
# B = LHS[nu+1:end, 1:nu]
# BT = LHS[1:nu, nu+1:end]
# D = LHS[nu+1:end, nu+1:end]
# @assert nnz(D) == 0
# F = RHS[1:nu]
# G = RHS[nu+1:end]

# # use incomplete LU for preconditioning on A
# @time "A_ilu" A_ilu = ilu(A, τ=1e-5)
# # @time "A_lu" A_lu = lu(A)
# # @assert issymmetric(A)
# # @assert isposdef(A)
# # Ainv = LinearMap(x -> cg(A, x, Pl=A_ilu, reltol=1e-6), nu, nu)

# # # schur complement
# # B = LinearMap(B)
# # BT = LinearMap(BT)
# # S = B*Ainv*BT

# # iterate on full matrix with block preconditioning
# struct BlockPreconditioner{A,B}
#   A::A
#   B::B
# end
# ldiv!(y, P::BlockPreconditioner, x) = begin
#   ldiv!(y[1:nu], P.A, x[1:nu])
#   ldiv!(y[nu+1:end], P.B, x[nu+1:end])
# end
# BP = BlockPreconditioner(A_ilu, M_ilu)
# # BP = BlockPreconditioner(A_lu, M_lu)
# # @time "sol" sol, ch = gmres(LHS, RHS, Pl=P, verbose=true, log=true)
# @time (sol, stats) = Krylov.gmres(LHS, RHS, M=BP, ldiv=true, verbose=1)
# # LHS_ilu = ilu(LHS, τ=1e-5)
# # @time (sol, stats) = Krylov.gmres(LHS, RHS, M=LHS_ilu, ldiv=true, verbose=1)
# u_sol = sol[1:nu]
# p_sol = sol[nu+1:end]

# # @time (u_sol, p_sol, stats) = gpmr(BT, B, F, G, C=A_ilu, D=M_lu, gsp=true, verbose=1, ldiv=true)
# # Ainv = LinearMap(x -> IterativeSolvers.bicgstabl!(copy(x), A, x, Pl=A_ilu, verbose=true), nu, nu)
# # Sinv = LinearMap(x -> IterativeSolvers.cg(M, x, Pl=M_lu, verbose=true), np-1, np-1)
# # @time (u_sol, p_sol, stats) = gpmr(BT, B, F, G, C=Ainv, D=Sinv, gsp=true, verbose=1)
# # println(stats)

# # # solve for pressure
# # @time "p_rhs" p_rhs = B*(Ainv*F) - G
# # @time "p_sol" p_sol, ch = cg(S, p_rhs, Pl=M_ilu, reltol=1e-6, verbose=true, log=true)

# # # solve for velocity
# # u_rhs = F - BT*p_sol
# # @time "u_sol" u_sol = cg(A, u_rhs, Pl=A_ilu, reltol=1e-6)
# # println(maximum(abs.(u_sol)))

# # export to vtk
# ux.free_values .= u_sol[1:nx]
# uy.free_values .= u_sol[nx+1:nx+ny]
# uz.free_values .= u_sol[nx+ny+1:nx+ny+nz]
# p = FEFunction(P, p_sol)
# writevtk(Ω, "out/nonhydro2D", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])