using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
model = GmshDiscreteModel("bowl.msh")
# writevtk(model, "model")
# error()

# reference FE 
reffe_ux = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, 1; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot", "corners"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Y = MultiFieldFESpace([Vx, Vz])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0, 0])
Uz = TrialFESpace(Vz, [0, 0, 0])
X  = MultiFieldFESpace([Ux, Uz])

# triangulation and integration measure
degree = 2
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)
Γ = BoundaryTriangulation(model, tags=["top"])
dΓ = Measure(Γ, degree)

# gradients 
x = VectorValue(1.0, 0.0)
z = VectorValue(0.0, 1.0)
∂x(u) = x⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
δ = 0.1
H(x) = sqrt(2 - x^2) - 1
b(x) = δ*exp(-(x[2] + H(x[1]))/δ)

# bilinear and linear form
# a((ux, uz), (vx, vz)) = ∫( ∂z(vx)*∂z(∂z(ux)) + vz*∂x(ux) + vz*∂z(uz) )dΩ #- ∫( vx*∂z(∂z(ux)) )dΓ
a((ux, uz), (vx, vz)) = ∫( ∂z(vx)*Δ(ux) + vz*∂x(ux) + vz*∂z(uz) )dΩ - ∫( vx*Δ(ux) )dΓ
l((vx, vz)) = ∫( b*vz )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ux, uz = solve(op)

# export to vtk
writevtk(Ωₕ, "results", cellfields=["ux"=>ux, "uz"=>uz])
