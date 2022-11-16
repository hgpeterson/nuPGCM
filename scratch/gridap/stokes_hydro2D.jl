using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
model = GmshDiscreteModel("bowl2D.msh")
writevtk(model, "model")
# error()

# reference FE 
reffe_ux = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, 1; space=:P)
reffe_p  = ReferenceFE(lagrangian, Float64, 0; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bottom"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bottom", "surface"])
Q  = TestFESpace(model, reffe_p,  conformity=:L2, constraint=:zeromean)
Y = MultiFieldFESpace([Vx, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0])
Uz = TrialFESpace(Vz, [0, 0])
P  = TrialFESpace(Q)
X  = MultiFieldFESpace([Ux, Uz, P])

# triangulation and integration measure
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# gradients 
x = VectorValue(1.0, 0.0)
z = VectorValue(0.0, 1.0)
∂x(u) = x⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
# δ = 0.1
# H(x) = sqrt(2 - x[1]^2) - 1
# b(x) = δ*exp(-(x[2] + H(x))/δ)
b(x) = x[1]

# bilinear and linear form
a((ux, uz, p), (vx, vz, q)) = ∫( ∂z(vx)*∂z(ux) - ∂x(vx)*p - ∂z(vz)*p + q*∂x(ux) + q*∂z(uz) )dΩ
l((vx, vz, q)) = ∫( b*vz )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ux, uz, p = solve(op)

# export to vtk
writevtk(Ω, "stokes_hydro2D", cellfields=["ux"=>ux, "uz"=>uz, "p"=>p])