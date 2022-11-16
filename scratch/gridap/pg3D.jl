using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
model = GmshDiscreteModel("bowl3D.msh")
writevtk(model, "model")
# error()

# reference FE 
reffe_ux = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_uy = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_uz = ReferenceFE(lagrangian, Float64, 1; space=:P)
reffe_p  = ReferenceFE(lagrangian, Float64, 0; space=:P)

# test FESpaces
Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bottom"])
Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bottom"])
Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bottom", "surface"])
Q  = TestFESpace(model, reffe_p,  conformity=:L2, constraint=:zeromean)
Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Vx, [0])
Uy = TrialFESpace(Vy, [0])
Uz = TrialFESpace(Vz, [0, 0])
P  = TrialFESpace(Q)
X  = MultiFieldFESpace([Ux, Uy, Uz, P])

# triangulation and integration measure
degree = 4
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

# gradients 
x = VectorValue(1.0, 0.0, 0.0)
y = VectorValue(0.0, 1.0, 0.0)
z = VectorValue(0.0, 0.0, 1.0)
∂x(u) = x⋅∇(u)
∂y(u) = y⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
# δ = 0.1
# H(x) = sqrt(2 - x[1]^2 - x[2]^2) - 1
# b(x) = δ*exp(-(x[3] + H(x))/δ)
b(x) = x[1]

# bilinear and linear form
# ε² = 1e-1
ε² = 1
a((ux, uy, uz, p), (vx, vy, vz, q)) = ∫( ε²*(∂z(ux)*∂z(vx) + ∂z(uy)*∂z(vy)) + 
                                         uy*vx - ux*vy + 
                                        -p*(∂x(vx) + ∂y(vy) + ∂z(vz)) + 
                                         (∂x(ux) + ∂y(uy) + ∂z(uz))*q )dΩ
l((vx, vy, vz, q)) = ∫( b*vz )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ux, uy, uz, p = solve(op)

# export to vtk
writevtk(Ω, "pg3D", cellfields=["ux"=>ux, "uy"=>uy, "uz"=>uz, "p"=>p])