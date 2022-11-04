using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
# model = GmshDiscreteModel("bowl.msh")
model = GmshDiscreteModel("bowl1.msh")
# writevtk(model, "model")
# error()

# reference FE 
order = 2
reffe_ωx = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_ωy = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_χx = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_χy = ReferenceFE(lagrangian, Float64, order; space=:P)

# test FESpaces
Tx = TestFESpace(model, reffe_ωx, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Ty = TestFESpace(model, reffe_ωy, conformity=:H1, dirichlet_tags=["top"])
Ψx = TestFESpace(model, reffe_χx, conformity=:H1, dirichlet_tags=["top"])
Ψy = TestFESpace(model, reffe_χy, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Y = MultiFieldFESpace([Tx, Ty, Ψx, Ψy])

# trial FESpaces with Dirichlet values
Wx = TrialFESpace(Tx, [0, 0, 0])
Wy = TrialFESpace(Ty, [0])
Xx = TrialFESpace(Ψx, [0])
Xy = TrialFESpace(Ψy, [0, 0, 0])
X  = MultiFieldFESpace([Wx, Wy, Xx, Xy])

# triangulation and integration measure
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

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
ε² = 1e-3
a((ωx, ωy, χx, χy), (τx, τy, ψx, ψy)) = ∫( ε²*∂z(ωx)*∂z(τx) - ωy*τx + 
                                          -ε²*∂z(ωy)*∂z(τy) - ωx*τy + #multiplied by -1 to get +bx 
                                           ∂z(χx)*∂z(ψx) - ωx*ψx +
                                           ∂z(χy)*∂z(ψy) - ωy*ψy
                                           )dΩ
l((τx, τy, ψx, ψy)) = ∫( bx*τy )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ωx, ωy, χx, χy = solve(op)

# compute velocities
ux = -∂z(χy)
uy = ∂z(χx)
uz = ∂x(χy)

# export to vtk
writevtk(Ω, "results", cellfields=["ωx"=>ωx, "ωy"=>ωy, "χx"=>χx, "χy"=>χy,
                                   "ux"=>ux, "uy"=>uy, "uz"=>uz])