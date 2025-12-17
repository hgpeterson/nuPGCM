using Gridap
using Gridap.ReferenceFEs
using GridapGmsh
using Gmsh
using Printf

# constructed solution
u(x) = VectorValue( sin(π*x[1])*cos(π*x[2]), -x[2]*x[1]^2 )
p(x) = exp(x[1])*x[2]^2
f(x) = -Δ(u)(x) + ∇(p)(x)
g(x) = (∇⋅u)(x)
∇u(x) = ∇(u)(x)

# for bubble workaround if using quad mesh (use `conformity=:L2`)
ReferenceFEs.Conformity(::GenericRefFE{Bubble}, ::Symbol) = L2Conformity()

function generate_mesh(shape=:TRI; h=1/2^3, L=1)
    if shape ∉ [:TRI, :QUAD]
        throw(ArgumentError("`shape` must be either :TRI or :QUAD"))
    end
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("mesh")
    gmsh.model.geo.addPoint(0, 0, 0)
    gmsh.model.geo.addPoint(L, 0, 0)
    gmsh.model.geo.addPoint(L, L, 0) 
    gmsh.model.geo.addPoint(0, L, 0) 
    gmsh.model.geo.addLine(1, 2)
    gmsh.model.geo.addLine(2, 3)
    gmsh.model.geo.addLine(3, 4)
    gmsh.model.geo.addLine(4, 1)
    gmsh.model.geo.addCurveLoop(1:4, 1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(0, [1], 1, "tag_1")
    gmsh.model.addPhysicalGroup(0, [2], 2, "dirichlet")
    gmsh.model.addPhysicalGroup(0, 3:4, 3, "neumann")
    gmsh.model.addPhysicalGroup(1, [1], 2, "dirichlet")
    gmsh.model.addPhysicalGroup(1, 2:4, 3, "neumann")
    gmsh.model.addPhysicalGroup(2, [1], 4, "interior")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    if shape == :QUAD
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)
    end
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()
end

function gmsh_test()
    model = GmshDiscreteModel("mesh.msh")

    reffe_u = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    reffe_b = ReferenceFE(bubble, VectorValue{2, Float64})
    reffe_p = ReferenceFE(lagrangian, Float64, 1)

    V = TestFESpace(model, reffe_u, dirichlet_tags=["tag_1", "dirichlet"], conformity=:H1)
    R = TestFESpace(model, reffe_b)
    # R = TestFESpace(model, reffe_b, conformity=:L2)  # for quad mesh
    Q = TestFESpace(model, reffe_p, dirichlet_tags="tag_1", conformity=:H1)

    U = TrialFESpace(V, u)
    B = TrialFESpace(R)
    P = TrialFESpace(Q, p)

    Y = MultiFieldFESpace([V, R, Q])
    X = MultiFieldFESpace([U, B, P])

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model, tags="neumann")
    n_Γ = get_normal_vector(Γ)

    degree = 4
    dΩ = Measure(Ω, degree)
    dΓ = Measure(Γ, degree)

    a((u, p), (v, q)) = ∫(∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u))*dΩ
    l((v, q)) = ∫(v⋅f + q*g)*dΩ + ∫(v⋅(n_Γ⋅∇u) - (n_Γ⋅v)*p)*dΓ

    mini_a((u, b, p), (v, r, q)) = a((u + b, p), (v+r, q))
    mini_l((v, r, q)) = l((v+r, q))

    op = AffineFEOperator(mini_a, mini_l, X, Y)

    uch, ubh, ph = solve(op)
    uh = ubh + uch

    writevtk(Ω, "sol", cellfields=["u"=>u, "uh"=>uh, "p"=>p, "ph"=>ph])
    @info "sol.vtu"

    eu = u - uh
    ep = p - ph

    l2(u) = sqrt(sum(∫(u⊙u)*dΩ))
    h1(u) = sqrt(sum(∫(u⊙u + ∇(u)⊙∇(u))*dΩ))

    eu_h1 = h1(eu) 
    ep_l2 = l2(ep)
    @printf("Error = %0.5e\n", eu_h1 + ep_l2)
end

generate_mesh(; L=1, h=1/2^3)
gmsh_test()

# tris:
# h = 1/2^3: 3.63816e-01
# h = 1/2^4: 1.68337e-01
# h = 1/2^5: 7.86498e-02
# h = 1/2^6: 3.80200e-02

# quads:
# h = 1/2^3: 1.51591e-01
# h = 1/2^4: 6.23113e-02
# h = 1/2^5: 2.92883e-02
# h = 1/2^6: 1.40128e-02