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
    spaces = Spaces(mesh::Mesh, u_diri, v_diri, w_diri, b_diri; order=2)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.

`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, v, w, p) while the `B`s are single-field spaces for 
buoyancy.
"""
function Spaces(mesh::Mesh, u_diri, v_diri, w_diri, b_diri; order=2)
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
    u_diri_tags = collect(keys(u_diri))
    v_diri_tags = collect(keys(v_diri))
    w_diri_tags = collect(keys(w_diri))
    b_diri_tags = collect(keys(b_diri))
    U_test = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=(length(u_diri_tags) > 0) ? u_diri_tags : Int[])
    V_test = TestFESpace(model, reffe_v, conformity=:H1, dirichlet_tags=(length(v_diri_tags) > 0) ? v_diri_tags : Int[])
    W_test = TestFESpace(model, reffe_w, conformity=:H1, dirichlet_tags=(length(w_diri_tags) > 0) ? w_diri_tags : Int[])
    P_test = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    X_test = MultiFieldFESpace([U_test, V_test, W_test, P_test])
    B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=(length(b_diri_tags) > 0) ? b_diri_tags : Int[])
    ν_test = TestFESpace(model, reffe_ν, conformity=:H1)
    κ_test = TestFESpace(model, reffe_κ, conformity=:H1)

    # trial FESpaces with Dirichlet values
    u_diri_vals = collect(values(u_diri))
    v_diri_vals = collect(values(v_diri))
    w_diri_vals = collect(values(w_diri))
    b_diri_vals = collect(values(b_diri))
    if length(u_diri_vals) > 0
        U_trial = TrialFESpace(U_test, u_diri_vals)
    else
        U_trial = TrialFESpace(U_test)
    end
    if length(v_diri_vals) > 0
        V_trial = TrialFESpace(V_test, v_diri_vals)
    else
        V_trial = TrialFESpace(V_test)
    end
    if length(w_diri_vals) > 0
        W_trial = TrialFESpace(W_test, w_diri_vals)
    else
        W_trial = TrialFESpace(W_test)
    end
    P_trial = TrialFESpace(P_test)
    X_trial = MultiFieldFESpace([U_trial, V_trial, W_trial, P_trial])
    if length(b_diri_vals) > 0
        B_trial = TrialFESpace(B_test, b_diri_vals)
    else
        B_trial = TrialFESpace(B_test)
    end
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