using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
model = GmshDiscreteModel("bowl.msh")
# writevtk(model, "model")
# error()

# reference FE 
order = 2
reffe_u = ReferenceFE(raviart_thomas, Float64, order)

# test FESpaces
V = TestFESpace(model, reffe_u, conformity=:HDiv, dirichlet_tags=["top", "bot", "corners"])

# trial FESpaces with Dirichlet values
U = TrialFESpace(V, [0, 0, 0])

# triangulation and integration measure
degree = 2*order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ, degree)

# forcing
δ = 0.1
H(x) = sqrt(2 - x^2) - 1
Hx(x) = -x/sqrt(2 - x^2)
bx(x) = -(1 + Hx(x[1]))*exp(-(x[2] + H(x[1]))/δ)

# bilinear and linear form
a(u, v) = ∫( Δu*Δv )dΩ 
l(v) = ∫( bx*v )dΩ

# affine FE operator
op = AffineFEOperator(a, l, U, V)

# solve
u = solve(op)

# export to vtk
writevtk(Ωₕ, "results", cellfields=["u"=>u])
