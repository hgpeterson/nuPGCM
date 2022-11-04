using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
model = GmshDiscreteModel("bowl.msh")
# writevtk(model, "model")
# error()

# reference FE 
order = 2
reffe_u = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_w = ReferenceFE(lagrangian, Float64, order; space=:P)

# test FESpaces
Φ = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Ψ = TestFESpace(model, reffe_w, conformity=:H1, constraint=:zeromean)
Y = MultiFieldFESpace([Φ, Ψ])

# trial FESpaces with Dirichlet values
U = TrialFESpace(Φ, [0, 0, 0])
W = TrialFESpace(Ψ)
X = MultiFieldFESpace([U, W])

# triangulation and integration measure
degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)

# forcing
δ = 0.1
H(x) = sqrt(2 - x^2) - 1
bx(x) = x[1]/sqrt(2 - x[1]^2)*exp(-(x[2] + H(x[1]))/δ)

# vertical derivative
x = VectorValue(1.0, 0.0)
z = VectorValue(0.0, 1.0)
∂x(u) = x⋅∇(u)
∂z(u) = z⋅∇(u)

# bilinear and linear form
a((u, w), (ϕ, ψ)) = ∫( w*ϕ - ∂z(u)*∂z(ϕ) + ∂z(w)*∂z(ψ) )dΩ 
l((ϕ, ψ)) = ∫( bx*ψ )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
u, w = solve(op)

# export to vtk
writevtk(Ωₕ, "results", cellfields=["u"=>u, "w"=>w])
