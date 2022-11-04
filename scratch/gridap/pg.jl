using Gridap
using GridapGmsh
using Gmsh: gmsh
using PyPlot

pygui(false)
plt.style.use("../nuPGCM/plots.mplstyle")
plt.close("all")

# model
model = GmshDiscreteModel("bowl.msh")
# writevtk(model, "model")
# error()

# reference FE 
reffe_ux = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_uy = ReferenceFE(lagrangian, Float64, 1; space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, 1; space=:P)
reffe_p  = ReferenceFE(lagrangian, Float64, 0; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot", "corners"])
Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot", "corners"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Q  = TestFESpace(model, reffe_p,  conformity=:L2, constraint=:zeromean)
# Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0, 0])
Uy = TrialFESpace(Vy, [0, 0])
Uz = TrialFESpace(Vz, [0, 0, 0])
P  = TrialFESpace(Q)
X  = MultiFieldFESpace([Ux, Uy, Uz, P])

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
ε² = 1e-3
a((ux, uy, uz, p), (vx, vy, vz, q)) = ∫( ε²*∂z(vx)*∂z(ux) + ε²*∂z(vy)*∂z(uy) + uy*vx - ux*vy - ∂x(vx)*p - ∂z(vz)*p + q*∂x(ux) + q*∂z(uz) )dΩ
l((vx, vy, vz, q)) = ∫( b*vz )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ux, uy, uz, p = solve(op)

# export to vtk
writevtk(Ωₕ, "results", cellfields=["ux"=>ux, "uy"=>uy, "uz"=>uz, "p"=>p])
