struct Spaces{X, Y, B, C}
    X_trial::X # trial space for [u, p]
    X_test::Y  # test space for [u, p]
    B_trial::B # trial space for buoyancy
    B_test::C  # test space for buoyancy
end

"""
    spaces = Spaces(model; order=2)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.
`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, p) while the `B`s are single-field spaces for buoyancy.
"""
function Spaces(model; order=2)
    # reference FE 
    reffe_u = ReferenceFE(lagrangian, VectorValue{3, Float64}, order;   space=:P)
    reffe_p = ReferenceFE(lagrangian, Float64,                 order-1; space=:P)
    reffe_b = ReferenceFE(lagrangian, Float64,                 order;   space=:P)

    # test FESpaces
    U_test = TestFESpace(model, reffe_u, conformity=:H1, 
                         dirichlet_tags=["bot", "sfc"], 
                         dirichlet_masks=[(true, true, true), (false, false, true)])
    P_test = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=["sfc"])
    X_test = MultiFieldFESpace([U_test, P_test])

    # trial FESpaces with Dirichlet values
    u_bot(x) = VectorValue(0.0, 0.0, 0.0)
    u_sfc(x) = VectorValue(0.0, 0.0, 0.0)
    U_trial = TrialFESpace(U_test, [u_bot, u_sfc])
    P_trial = TrialFESpace(P_test)
    B_trial = TrialFESpace(B_test, [0])
    X_trial = MultiFieldFESpace([U_trial, P_trial])

    return Spaces(X_trial, X_test, B_trial, B_test)
end