struct Spaces{X1, X2, B1, B2, N1, N2, K1, K2, BD}
    X_trial::X1  # trial space for [u, v, w, p]
    X_test::X2   # test space for [u, v, w, p]
    B_trial::B1  # trial space for buoyancy
    B_test::B2   # test space for buoyancy
    ν_trial::N1  # trial space for viscosity
    ν_test::N2   # test space for viscosity
    κ_trial::K1  # trial space for vertical diffusivity
    κ_test::K2   # test space for vertical diffusivity
    b_diri::BD   # FEFunction that is b₀ on the dirichlet boundary and 0 elsewhere
end

"""
    spaces = Spaces(model, b₀; order=2)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.

`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, v, w, p) while the `B`s are single-field spaces for 
buoyancy. `b₀` a function for the buoyancy surface boundary condition.
"""
# function Spaces(mesh::Mesh, b_surface; order=2)
function Spaces(mesh::Mesh, b_surface, b_basin; order=2)
    model = mesh.model

    # reference FE 
    reffe_u = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_v = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_w = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_p = ReferenceFE(lagrangian, Float64, order-1; space=:P)
    reffe_b = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_ν = ReferenceFE(lagrangian, Float64, 1; space=:P)
    reffe_κ = ReferenceFE(lagrangian, Float64, 1; space=:P)

    # test FESpaces
    # U_test = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["bottom", "coastline"])
    # V_test = TestFESpace(model, reffe_v, conformity=:H1, dirichlet_tags=["bottom", "coastline"])
    # W_test = TestFESpace(model, reffe_w, conformity=:H1, dirichlet_tags=["bottom", "coastline", "surface"])
    U_test = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["bottom", "coastline", "basin bottom"])
    V_test = TestFESpace(model, reffe_v, conformity=:H1, dirichlet_tags=["bottom", "coastline", "basin bottom"])
    W_test = TestFESpace(model, reffe_w, conformity=:H1, dirichlet_tags=["bottom", "coastline", "surface", "basin bottom", "basin"])
    P_test = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    X_test = MultiFieldFESpace([U_test, V_test, W_test, P_test])
    # B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=["coastline", "surface"])
    B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=["coastline", "surface", "basin", "basin bottom"])
    ν_test = TestFESpace(model, reffe_ν, conformity=:H1)
    κ_test = TestFESpace(model, reffe_κ, conformity=:H1)

    # trial FESpaces with Dirichlet values
    # U_trial = TrialFESpace(U_test, [0, 0])
    # V_trial = TrialFESpace(V_test, [0, 0])
    # W_trial = TrialFESpace(W_test, [0, 0, 0])
    U_trial = TrialFESpace(U_test, [0, 0, 0])
    V_trial = TrialFESpace(V_test, [0, 0, 0])
    W_trial = TrialFESpace(W_test, [0, 0, 0, 0, 0])
    P_trial = TrialFESpace(P_test)
    X_trial = MultiFieldFESpace([U_trial, V_trial, W_trial, P_trial])
    # B_trial = TrialFESpace(B_test, [b_surface, b_surface])
    B_trial = TrialFESpace(B_test, [b_surface, b_surface, b_basin, b_basin])
    ν_trial = TrialFESpace(ν_test)
    κ_trial = TrialFESpace(κ_test)

    # a FEFunction that is b₀ on the dirichlet boundary and 0 elsewhere
    # (needed for assembling matrices)
    b_diri = interpolate(0, B_trial)

    return Spaces(X_trial, X_test, B_trial, B_test, ν_trial, ν_test, κ_trial, κ_test, b_diri)
end

function get_U_V_W_P(spaces::Spaces)
    U = spaces.X_trial[1]
    V = spaces.X_trial[2]
    W = spaces.X_trial[3]
    P = spaces.X_trial[4]
    return U, V, W, P
end