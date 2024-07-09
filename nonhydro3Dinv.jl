# using NonhydroPG
# using Gridap
# using GridapGmsh
# using Gmsh: gmsh
# using IncompleteLU
# using IterativeSolvers
# using LinearMaps
# using SparseArrays
# using Preconditioners
# using LinearAlgebra
# import LinearAlgebra: ldiv!, \

# # model
# model = GmshDiscreteModel("bowl3D_0.05.msh")
# # writevtk(model, "model")

# # reference FE 
# order = 2
# reffe_ux = ReferenceFE(lagrangian, Float64, order;   space=:P)
# reffe_uy = ReferenceFE(lagrangian, Float64, order;   space=:P)
# reffe_uz = ReferenceFE(lagrangian, Float64, order;   space=:P)
# reffe_p  = ReferenceFE(lagrangian, Float64, order-1; space=:P)

# # test FESpaces
# # Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot"])
# # Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot"])
# Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot", "sfc"])
# Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot", "sfc"])
# Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
# Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
# Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

# # trial FESpaces with Dirichlet values
# # Ux = TrialFESpace(Vx, [0])
# # Uy = TrialFESpace(Vy, [0])
# Ux = TrialFESpace(Vx, [0, 1])
# Uy = TrialFESpace(Vy, [0, 0])
# Uz = TrialFESpace(Vz, [0, 0])
# P  = TrialFESpace(Q)
# X  = MultiFieldFESpace([Ux, Uy, Uz, P])
# nx = Ux.space.nfree
# ny = Uy.space.nfree
# nz = Uz.space.nfree
# nu = nx + ny + nz
# np = P.space.space.nfree
# println("N = ", nu+np-1, " (", nu, " + ", np-1, ")")

# # initialize vectors
# ux = interpolate_everywhere(0, Ux)
# uy = interpolate_everywhere(0, Uy)
# uz = interpolate_everywhere(0, Uz)
# p  = interpolate_everywhere(0, P)
# ux.dirichlet_values .= Ux.dirichlet_values
# uy.dirichlet_values .= Uy.dirichlet_values
# uz.dirichlet_values .= Uz.dirichlet_values

# # triangulation and integration measure
# degree = order^2
# Ω = Triangulation(model)
# dΩ = Measure(Ω, degree)

# # gradients 
# ∂x(u) = VectorValue(1.0, 0.0, 0.0)⋅∇(u)
# ∂y(u) = VectorValue(0.0, 1.0, 0.0)⋅∇(u)
# ∂z(u) = VectorValue(0.0, 0.0, 1.0)⋅∇(u)

# # forcing
# H(x) = sqrt(2 - x[1]^2 - x[2]^2) - 1
# ν(x) = 1

# # params
# ε² = 1
# println("δ = ", √(2ε²))
# pts, conns = get_p_t(model)
# h1 = [norm(pts[conns[i, 1], :] - pts[conns[i, 2], :]) for i ∈ axes(conns, 1)]
# h2 = [norm(pts[conns[i, 2], :] - pts[conns[i, 3], :]) for i ∈ axes(conns, 1)]
# h3 = [norm(pts[conns[i, 3], :] - pts[conns[i, 4], :]) for i ∈ axes(conns, 1)]
# h4 = [norm(pts[conns[i, 4], :] - pts[conns[i, 1], :]) for i ∈ axes(conns, 1)]
# hmin = minimum([h1; h2; h3; h4])
# hmax = maximum([h1; h2; h3; h4])
# println("hmin = ", hmin)
# println("hmax = ", hmax)
# γ = 1
# f(x) = 1

# # LHS matrix
# a((ux, uy, uz, p), (vx, vy, vz, q)) = 
#     # ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν - uy*vx*f + ∂x(p)*vx +
#     #    γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ux*vy*f + ∂y(p)*vy +
#     #  γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
#     #                                                               ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
#     ∫( γ*ε²*∂x(ux)*∂x(vx)*ν +   γ*ε²*∂y(ux)*∂y(vx)*ν +   ε²*∂z(ux)*∂z(vx)*ν + ∂x(p)*vx +
#        γ*ε²*∂x(uy)*∂x(vy)*ν +   γ*ε²*∂y(uy)*∂y(vy)*ν +   ε²*∂z(uy)*∂z(vy)*ν + ∂y(p)*vy +
#      γ^2*ε²*∂x(uz)*∂x(vz)*ν + γ^2*ε²*∂y(uz)*∂y(vz)*ν + γ*ε²*∂z(uz)*∂z(vz)*ν + ∂z(p)*vz +
#                                                         ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
#     # ∫( ε²*∂z(ux)*∂z(vx)*ν - f*uy*vx + ∂x(p)*vx +
#     #    ε²*∂z(uy)*∂z(vy)*ν + f*ux*vy + ∂y(p)*vy +
#     #  γ*ε²*∂z(uz)*∂z(vz)*ν +           ∂z(p)*vz +
#     #             ∂x(ux)*q + ∂y(uy)*q + ∂z(uz)*q )dΩ
# # @time "assemble LHS" LHS = assemble_matrix(a, X, Y)
# # @time "iLU" iLU = ilu(LHS, τ=1e-5)

# # initial guess
# # b0(x) = x[3] + 0.10*exp(-(x[3] + H(x))/0.10)
# b0(x) = 0
# l0((vx, vy, vz, q)) = ∫( b0*vz )dΩ
# # @time "assemble RHS 0" RHS0 = assemble_vector(l0, Y)
# # @time "solve 0" sol0, ch = bicgstabl(LHS, RHS0, Pl=iLU, verbose=true, log=true)

# # use Gridap operator for now because it seems to give a different RHS
# @time "assemble LHS and RHS0" op = AffineFEOperator(a, l0, X, Y)
# LHS = get_matrix(op.op)
# RHS0 = get_vector(op.op)

# # exact solve
# @time "solve" sol = LHS\RHS0
# ux.free_values .= sol[1:nx]
# uy.free_values .= sol[nx+1:nx+ny]
# uz.free_values .= sol[nx+ny+1:nx+ny+nz]
# p = FEFunction(P, sol[nx+ny+nz+1:end])
# writevtk(Ω, "out/nonhydro3D_true", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p])

# # perturbation
# # b1(x) = x[3] + 0.11*exp(-(x[3] + H(x))/0.11)
# b1(x) = 0
# l1((vx, vy, vz, q)) = ∫( b1*vz )dΩ
# @time "assemble RHS 1" RHS1 = assemble_vector(l1, Y)
# # sol1 = copy(sol0)
# # @time "solve 1" sol1, ch = bicgstabl!(sol1, LHS, RHS1, Pl=iLU, verbose=true, log=true)

# # assemble mass matrix for preconditioning
# a_m(p, q) = ∫( p*q )dΩ
# @time "assemble M" M = assemble_matrix(a_m, P, Q)
# # struct MassMatrixPreconditioner{M} <: Preconditioners.AbstractPreconditioner
# #     M::M
# # end
# # ldiv!(y, M::MassMatrixPreconditioner, x) = cg!(y, M.M, x, Pl=DiagonalPreconditioner(M.M))
# # ldiv!(M::MassMatrixPreconditioner, x) = cg!(x, M.M, x, Pl=DiagonalPreconditioner(M.M))
# # \(M::MassMatrixPreconditioner, x) = cg(M.M, x, Pl=DiagonalPreconditioner(M.M))
# # PM = MassMatrixPreconditioner(M)
# @time "lu(M)" PM = lu(M/ε²)

# set up block matrices: LHS = [A B; BT D] where D should be 0; RHS = [F; G]
A = LHS[1:nu, 1:nu]
B = LHS[nu+1:end, 1:nu]
BT = LHS[1:nu, nu+1:end]
D = LHS[nu+1:end, nu+1:end]
@assert nnz(D) == 0
F0 = RHS0[1:nu]
G0 = RHS0[nu+1:end]
# F1 = RHS1[1:nu]
# G1 = RHS1[nu+1:end]

# # use incomplete LU for preconditioning on A
# @time "ALU" ALU = ilu(A, τ=1e-5)
# # @time "issym" @assert issymmetric(A)
# # @time "isposdef" @assert isposdef(A)
# Ainv = LinearMap(x -> cg!(x, A, x, Pl=ALU, reltol=1e-6), nu, nu)

# # schur complement
# B = LinearMap(B)
# BT = LinearMap(BT)
# S = B*Ainv*BT

# solve for pressure
# @time "p_rhs 0" p_rhs0 = B*(Ainv*F0) - G0
# @time "p_rhs 0" p_rhs0 = B*(A\F0) - G0
# @time "p_sol 0" p_sol0, ch = cg(S, p_rhs0, Pl=PM, reltol=1e-6, verbose=true, log=true)
# @time "p_sol 0" p_sol0 = BT\(A*(B\p_rhs0))

# solve for velocity
u_rhs0 = F0 - BT*p_sol0
@time "u_sol 0" u_sol0 = cg(A, u_rhs0, Pl=ALU, reltol=1e-6)
# @time "u_sol 0" u_sol0 = A\u_rhs0

# export to vtk
ux.free_values .= u_sol0[1:nx]
uy.free_values .= u_sol0[nx+1:nx+ny]
uz.free_values .= u_sol0[nx+ny+1:nx+ny+nz]
p = FEFunction(P, p_sol0)
writevtk(Ω, "out/nonhydro3D", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b0])

# # solve for pressure
# @time "p_rhs 1" p_rhs1 = B*(Ainv*F1) - G1
# p_sol1 = copy(p_sol0)
# @time "p_sol 1" p_sol1, ch = cg!(p_sol1, S, p_rhs1, Pl=PM, verbose=true, log=true)

# # solve for velocity
# u_rhs1 = F1 - BT*p_sol1
# u_sol1 = copy(u_sol0)
# @time "u_sol 1" bicgstabl!(u_sol1, A, u_rhs1, Pl=ALU)

# # export to vtk
# ux.free_values .= u_sol1[1:nx]
# uy.free_values .= u_sol1[nx+1:nx+ny]
# uz.free_values .= u_sol1[nx+ny+1:nx+ny+nz]
# p = FEFunction(P, p_sol1)
# writevtk(Ω, "out/nonhydro3D", cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b1])