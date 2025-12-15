using Gridap
using Gridap.ReferenceFEs
using GridapGmsh
using Gmsh

# for bubble workaround if using quad mesh (use `conformity=:L2`)
ReferenceFEs.Conformity(::GenericRefFE{Bubble}, ::Symbol) = L2Conformity()

# Make sure `∇(b)` and `Δ(b)` are both continuous.
function b((x, y))
    x0 = floor(x)
    y0 = floor(y)
    ret = (x - x0) * (y - y0) * (x0 + 1 - x) * (y0 + 1 - y)
    sign = (-1)^(x0+y0)
    return 6^4 * sign * ret
end

u(x) = VectorValue(b(x) + 2*x[2], 3*b(x) + x[2] + 2*x[1])
p(x) = x[1] - 3*x[2]
f(x) = -Δ(u)(x) + ∇(p)(x)
g(x) = (∇⋅u)(x)
∇u(x) = ∇(u)(x)

################################################################################

function gridap_test()
    domain = (0, 3, 0, 3)
    partition = (3, 3)
    model = CartesianDiscreteModel(domain, partition)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "dirichlet", [1, 2, 5])
    add_tag_from_tags!(labels, "neumann", [3, 4, 6, 7, 8])

    reffe_u = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    reffe_b = ReferenceFE(bubble, VectorValue{2, Float64})
    reffe_p = ReferenceFE(lagrangian, Float64, 1)

    V = TestFESpace(model, reffe_u, labels = labels, dirichlet_tags = "dirichlet", conformity = :H1)
    R = TestFESpace(model, reffe_b)
    Q = TestFESpace(model, reffe_p, labels = labels, conformity = :H1, dirichlet_tags = "tag_1")

    U = TrialFESpace(V, u)
    B = TrialFESpace(R)
    P = TrialFESpace(Q, p)

    Y = MultiFieldFESpace([V, R, Q])
    X = MultiFieldFESpace([U, B, P])

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model, labels, tags = "neumann")
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

    eu = u - uh
    ep = p - ph

    l2(u) = sqrt(sum(∫(u⊙u)*dΩ))
    h1(u) = sqrt(sum(∫(u⊙u + ∇(u)⊙∇(u))*dΩ))

    @info eu_l2 = l2(eu)
    @info eu_h1 = h1(eu)
    @info ep_l2 = l2(ep)
end

################################################################################

function generate_mesh(; h=1, L=3)
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
    # gmsh.model.addPhysicalGroup(0, [1], 1, "tag_1")
    # gmsh.model.addPhysicalGroup(0, [2], 2, "dirichlet")
    # gmsh.model.addPhysicalGroup(0, 1:2, 2, "dirichlet")
    # gmsh.model.addPhysicalGroup(0, 3:4, 3, "neumann")
    # gmsh.model.addPhysicalGroup(1, [1], 2, "dirichlet")
    # gmsh.model.addPhysicalGroup(1, 2:4, 3, "neumann")
    # gmsh.model.addPhysicalGroup(2, [1], 4, "interior")
    gmsh.model.addPhysicalGroup(0, 1:4, 1, "dirichlet")
    gmsh.model.addPhysicalGroup(1, 1:4, 1, "dirichlet")
    gmsh.model.addPhysicalGroup(2, [1], 2, "interior")
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # rectangles
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # rectangles
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()
end

function gmsh_test()
    model = GmshDiscreteModel("mesh.msh")

    reffe_u = ReferenceFE(lagrangian, VectorValue{2, Float64}, 1)
    reffe_b = ReferenceFE(bubble, VectorValue{2, Float64})
    reffe_p = ReferenceFE(lagrangian, Float64, 1)

    # V = TestFESpace(model, reffe_u, dirichlet_tags=["tag_1", "dirichlet"], conformity=:H1)
    V = TestFESpace(model, reffe_u, dirichlet_tags="dirichlet", conformity=:H1)
    # R = TestFESpace(model, reffe_b)
    R = TestFESpace(model, reffe_b, conformity=:L2)  # for quad mesh
    # Q = TestFESpace(model, reffe_p, dirichlet_tags="tag_1", conformity=:H1)
    Q = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)

    U = TrialFESpace(V, u)
    B = TrialFESpace(R)
    P = TrialFESpace(Q, p)

    Y = MultiFieldFESpace([V, R, Q])
    X = MultiFieldFESpace([U, B, P])

    Ω = Triangulation(model)
    # Γ = BoundaryTriangulation(model, tags="neumann")
    # n_Γ = get_normal_vector(Γ)

    degree = 4
    dΩ = Measure(Ω, degree)
    # dΓ = Measure(Γ, degree)

    a((u, p), (v, q)) = ∫(∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u))*dΩ
    l((v, q)) = ∫(v⋅f + q*g)*dΩ #+ ∫(v⋅(n_Γ⋅∇u) - (n_Γ⋅v)*p)*dΓ

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

    @info eu_l2 = l2(eu)
    @info eu_h1 = h1(eu)
    @info ep_l2 = l2(ep)
end

################################################################################

function gmsh_test_TH()
    u(x) = VectorValue( x[1]^2 + 2*x[2]^2, -x[1]^2 )
    p(x) = x[1] + 3*x[2]
    f(x) = -Δ(u)(x) + ∇(p)(x)
    g(x) = (∇⋅u)(x)
    ∇u(x) = ∇(u)(x)

    model = GmshDiscreteModel("mesh.msh")

    order = 2

    reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_p = ReferenceFE(lagrangian,Float64,order-1)

    V = TestFESpace(model,reffe_u,dirichlet_tags="dirichlet",conformity=:H1)
    Q = TestFESpace(model,reffe_p,conformity=:H1)

    U = TrialFESpace(V,u)
    P = TrialFESpace(Q)

    Y = MultiFieldFESpace([V,Q])
    X = MultiFieldFESpace([U,P])

    Ω = Triangulation(model)
    Γ = BoundaryTriangulation(model,tags="neumann")
    n_Γ = get_normal_vector(Γ)

    degree = order
    dΩ = Measure(Ω,degree)
    dΓ = Measure(Γ,degree)

    a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )*dΩ

    l((v,q)) = ∫( v⋅f + q*g )*dΩ + ∫( v⋅(n_Γ⋅∇u) - (n_Γ⋅v)*p )*dΓ

    op = AffineFEOperator(a,l,X,Y)

    uh, ph = solve(op)

    eu = u - uh
    ep = p - ph

    l2(u) = sqrt(sum( ∫( u⊙u )*dΩ ))
    h1(u) = sqrt(sum( ∫( u⊙u + ∇(u)⊙∇(u) )*dΩ ))

    @info eu_l2 = l2(eu)
    @info eu_h1 = h1(eu)
    @info ep_l2 = l2(ep)
end

# gridap_test()
generate_mesh(; L=3, h=1)
gmsh_test()

# # passes for Taylor-Hood
# gmsh_test_TH()