using Gridap
using GridapGmsh
using Gmsh: gmsh

# model
model = GmshDiscreteModel("bowl3D.msh")
writevtk(model, "bowl3D")

# reference FE 
order = 2
reffe_ωx = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_ωy = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_χx = ReferenceFE(lagrangian, Float64, order; space=:P)
reffe_χy = ReferenceFE(lagrangian, Float64, order; space=:P)

# test FESpaces
Tx = TestFESpace(model, reffe_ωx, conformity=:H1, dirichlet_tags=["surface", "bottom"])
Ty = TestFESpace(model, reffe_ωy, conformity=:H1, dirichlet_tags=["surface"])
Ψx = TestFESpace(model, reffe_χx, conformity=:H1, dirichlet_tags=["surface"])
Ψy = TestFESpace(model, reffe_χy, conformity=:H1, dirichlet_tags=["surface", "bottom"])
Y = MultiFieldFESpace([Tx, Ty, Ψx, Ψy])

# trial FESpaces with Dirichlet values
Wx = TrialFESpace(Tx, [0, 0])
Wy = TrialFESpace(Ty, [0])
Xx = TrialFESpace(Ψx, [0])
Xy = TrialFESpace(Ψy, [0, 0])
X  = MultiFieldFESpace([Wx, Wy, Xx, Xy])

# triangulation and integration measure
degree = 2*order
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)
Γ = BoundaryTriangulation(model, tags=["bottom"])
dΓ = Measure(Γ, degree)

# gradients 
x = VectorValue(1.0, 0.0, 0.0)
y = VectorValue(0.0, 1.0, 0.0)
z = VectorValue(0.0, 0.0, 1.0)
∂x(u) = x⋅∇(u)
∂y(u) = y⋅∇(u)
∂z(u) = z⋅∇(u)

# forcing
δ = 0.1
H(x) = sqrt(2 - x[1]^2 - x[2]^2) - 1
bx(x) = x[1]/sqrt(2 - x[1]^2 - x[2]^2)*exp(-(x[3] + H(x))/δ)
by(x) = x[2]/sqrt(2 - x[1]^2 - x[2]^2)*exp(-(x[3] + H(x))/δ)
# bx(x) = 2
# by(x) = -3

# parameter
ε² = 1e-2
println("q⁻¹ = ", sqrt(2*ε²))
println("h   = ", 1/cbrt(size(model.grid.node_coordinates, 1)))

# bilinear and linear form
a((ωx, ωy, χx, χy), (τx, τy, ψx, ψy)) = ∫( ε²*∂z(ωx)*∂z(τx) - ωy*τx + 
                                          -ε²*∂z(ωy)*∂z(τy) - ωx*τy + #multiplied by -1 to get +bx 
                                          -∂z(χx)*∂z(ψx) + ωx*ψx +
                                          -∂z(χy)*∂z(ψy) + ωy*ψy)dΩ + 
                                        ∫(-ε²*∂z(ωx)*τx + ε²*∂z(ωy)*τy)dΓ
l((τx, τy, ψx, ψy)) = ∫( by*τx + bx*τy )dΩ

# affine FE operator
op = AffineFEOperator(a, l, X, Y)

# solve
ωx, ωy, χx, χy = solve(op)

# compute velocities
ux = -∂z(χy)
uy = ∂z(χx)
uz = ∂x(χy) - ∂y(χx)

# export to vtk
writevtk(Ω, "pg_vort_3D", cellfields=["ωx"=>ωx, "ωy"=>ωy, "χx"=>χx, "χy"=>χy,
                                      "ux"=>ux, "uy"=>uy, "uz"=>uz])