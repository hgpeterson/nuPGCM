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
reffe_τx = ReferenceFE(lagrangian, Float64, 1; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot", "corners"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Σx = TestFESpace(model, reffe_τx, conformity=:H1, dirichlet_tags=["top"])
Y = MultiFieldFESpace([Vx, Vz, Σx])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0, 0])
Uz = TrialFESpace(Vz, [0, 0, 0])
Tx = TrialFESpace(Σx, [0])
X  = MultiFieldFESpace([Ux, Uz, Tx])

# triangulation and integration measure
degree = 4
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γ = BoundaryTriangulation(model, tags=["bot", "corners"])
dΓ = Measure(Γ, degree)

# gradients 
x = VectorValue(1.0, 0.0)
z = VectorValue(0.0, 1.0)
∂x(u) = x⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
δ = 0.1
H(x) = sqrt(2 - x^2) - 1
bx(x) = x[1]/sqrt(2 - x[1]^2)*exp(-(x[2] + H(x[1]))/δ)

# bilinear and linear form
a((ux, uz, τx), (vx, vz, σx)) = ∫( -∂z(τx)*∂z(σx) + τx*vx - ∂z(ux)*vx + (∂x(ux) + ∂z(uz))*vz )dΩ + ∫( ∂z(τx)*σx )dΓ
l((vx, vz, σx)) = ∫( bx*σx )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ux, uz, τx = solve(op)

# export to vtk
writevtk(Ω, "results", cellfields=["ux"=>ux, "uz"=>uz, "τx"=>τx])