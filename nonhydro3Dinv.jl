using NonhydroPG
using Gridap
using GridapGmsh
using Gmsh: gmsh
using IncompleteLU
using Krylov
using LinearAlgebra
import LinearAlgebra: ldiv!, mul!

# model
model = GmshDiscreteModel("bowl3D_0.05.msh")
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
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0])
Uy = TrialFESpace(Vy, [0])
Uz = TrialFESpace(Vz, [0, 0])
P  = TrialFESpace(Q)
X  = MultiFieldFESpace([Ux, Uy, Uz, P])
nx = Ux.space.nfree
ny = Uy.space.nfree
nz = Uz.space.nfree
nu = nx + ny + nz
np = P.space.space.nfree
N = nu + np - 1
println("N = ", N, " (", nu, " + ", np-1, ")")

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

# forcing
H(x) = sqrt(2 - x[1]^2 - x[2]^2) - 1
ν(x) = 1

# params
ε² = 1
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
γ = 1
f(x) = 1

# LHS matrix
a((ux, uy, uz, p), (vx, vy, vz, q)) = 
    # ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
    #    γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
    #  γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
    #                                                               ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
    ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν + ∂x(p)*vx +
       γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ∂y(p)*vy +
     γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν + ∂z(p)*vz +
                                                                  ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
# @time "assemble LHS" LHS = assemble_matrix(a, X, Y)
@time "iLU" LHS_ilu = ilu(LHS, τ=1e-5)

# Krylov solver
solver = GmresSolver(N, N, 20, Vector{eltype(LHS)})

# initial guess
b0(x) = x[3] + 0.10*exp(-(x[3] + H(x))/0.10)
l0((vx, vy, vz, q)) = ∫( b0*vz )dΩ
@time "assemble RHS 0" RHS0 = assemble_vector(l0, Y)
# @time "solve 0" Krylov.solve!(solver, LHS, RHS0, M=LHS_ilu, verbose=1, ldiv=true)
# ux.free_values .= solver.x[1:nx]
# uy.free_values .= solver.x[nx+1:nx+ny]
# uz.free_values .= solver.x[nx+ny+1:nx+ny+nz]
# p = FEFunction(P, solver.x[nx+ny+nz+1:end])
# writevtk(Ω, "out/nonhydro3D", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b0])

# perturbation
b1(x) = x[3] + 0.11*exp(-(x[3] + H(x))/0.11)
l1((vx, vy, vz, q)) = ∫( b1*vz )dΩ
@time "assemble RHS 1" RHS1 = assemble_vector(l1, Y)
# @time "solve 1" Krylov.solve!(solver, LHS, RHS1, solver.x, M=LHS_ilu, verbose=1, ldiv=true)

# assemble mass matrix for preconditioning
a_m(p, q) = ∫( p*q )dΩ
@time "assemble M" M = assemble_matrix(a_m, P, Q)
@time "lu(M)" M_lu = lu(M/ε²)

# set up block matrices: LHS = [A B; BT D] where D should be 0; RHS = [F; G]
A = LHS[1:nu, 1:nu]
B = LHS[nu+1:end, 1:nu]
BT = LHS[1:nu, nu+1:end]

# use incomplete LU for preconditioning on A
@time "A_ilu" A_ilu = ilu(A, τ=1e-5)

# P⁻¹ = [A⁻¹  0  ]
#       [0    B⁻¹]
struct DiagonalBlockPreconditioner{A,B}
  A::A
  B::B
end
function ldiv!(y, P::DiagonalBlockPreconditioner, x)
  y[1:nu] = P.A\x[1:nu]
  y[nu+1:end] = P.B\x[nu+1:end]
  return y
end
DBP = DiagonalBlockPreconditioner(A_ilu, M_lu)
@time "solve with DBP" Krylov.solve!(solver, LHS, RHS0, M=DBP, ldiv=true, verbose=1)

# P⁻¹ = [A⁻¹      0  ]
#       [S⁻¹BA⁻¹  S⁻¹]
struct LowerTriangularBlockPreconditioner{A,B,S}
  A::A
  B::B
  S::S
end
function ldiv!(y, P::LowerTriangularBlockPreconditioner, x)
  unew = P.A\x[1:nu]
  y[1:nu] = unew
  y[nu+1:end] = P.S\(P.B*(unew) - x[nu+1:end])
  return y
end
LTBP = LowerTriangularBlockPreconditioner(A_ilu, B, M_lu)
@time "solve with LTBP" Krylov.solve!(solver, LHS, RHS0, M=LTBP, ldiv=true, verbose=1)
show(solver.stats)

@time "solve with ILU" Krylov.solve!(solver, LHS, RHS0, M=LHS_ilu, verbose=1, ldiv=true)