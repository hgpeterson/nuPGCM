struct Spaces{X, Y, B, C}
    X_trial::X # trial space for [u, v, w, p]
    X_test::Y  # test space for [u, v, w, p]
    B_trial::B # trial space for buoyancy
    B_test::C  # test space for buoyancy
end

"""
    spaces = Spaces(model; order=2)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.
`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, v, w, p) while the `B`s are single-field spaces for 
buoyancy.
"""
function Spaces(model; order=2)
    # reference FE 
    reffe_u = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_v = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_w = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_p = ReferenceFE(lagrangian, Float64, order-1; space=:P)
    reffe_b = ReferenceFE(lagrangian, Float64, order;   space=:P)

    # test FESpaces
    U_test = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["bot"])
    V_test = TestFESpace(model, reffe_v, conformity=:H1, dirichlet_tags=["bot"])
    W_test = TestFESpace(model, reffe_w, conformity=:H1, dirichlet_tags=["bot", "sfc"])
    P_test = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=["sfc"])
    X_test = MultiFieldFESpace([U_test, V_test, W_test, P_test])

    # trial FESpaces with Dirichlet values
    U_trial = TrialFESpace(U_test, [0])
    V_trial = TrialFESpace(V_test, [0])
    W_trial = TrialFESpace(W_test, [0, 0])
    P_trial = TrialFESpace(P_test)
    B_trial = TrialFESpace(B_test, [0])
    X_trial = MultiFieldFESpace([U_trial, V_trial, W_trial, P_trial])

    return Spaces(X_trial, X_test, B_trial, B_test)
end

function get_U_V_W_P(spaces::Spaces)
    U = spaces.X_trial[1]
    V = spaces.X_trial[2]
    W = spaces.X_trial[3]
    P = spaces.X_trial[4]
    return U, V, W, P
end