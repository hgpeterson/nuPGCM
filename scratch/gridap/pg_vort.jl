using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
# model = GmshDiscreteModel("bowl.msh")
model = GmshDiscreteModel("bowl1.msh")
# writevtk(model, "model")
# error()

# reference FE 
reffe_ωx = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_ωy = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_χx = ReferenceFE(lagrangian, Float64, 2; space=:P)
reffe_χy = ReferenceFE(lagrangian, Float64, 2; space=:P)

# test FESpaces
Ox = TestFESpace(model, reffe_ωx, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Oy = TestFESpace(model, reffe_ωy, conformity=:H1, dirichlet_tags=["top"])
Xx = TestFESpace(model, reffe_χx, conformity=:H1, dirichlet_tags=["top"])
Xy = TestFESpace(model, reffe_χy, conformity=:H1, dirichlet_tags=["top", "bot", "corners"])
Y = MultiFieldFESpace([Ox, Oy, Xx, Xy])

# trial FESpaces with Dirichlet values
Ux = TrialFESpace(Ox, [0, 0, 0])
Uy = TrialFESpace(Oy, [0])
Cx = TrialFESpace(Xx, [0])
Cy = TrialFESpace(Xy, [0, 0, 0])
X  = MultiFieldFESpace([Ux, Uy, Cx, Cy])

# triangulation and integration measure
degree = 4
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
ε² = 1e-5
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