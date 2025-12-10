struct Spaces{X1, X2, B1, B2, BD}
    X_trial::X1  # trial space for [u, v, w, p]
    X_test::X2   # test space for [u, v, w, p]
    B_trial::B1  # trial space for buoyancy
    B_test::B2   # test space for buoyancy
    b_diri::BD   # FEFunction that is b₀ on the dirichlet boundary and 0 elsewhere
end

function Base.summary(spaces::Spaces)
    t = typeof(spaces)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, spaces::Spaces)
    println(io, summary(spaces), ":")
    println(io, "├── X_trial: ", summary(spaces.X_trial))
    println(io, "├── X_test: ", summary(spaces.X_test))
    println(io, "├── B_trial: ", summary(spaces.B_trial))
    println(io, "├── B_test: ", summary(spaces.B_test))
      print(io, "└── b_diri: ", summary(spaces.b_diri))
end

"""
    spaces = Spaces(mesh::Mesh, u_diri, v_diri, w_diri, b_diri)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.

`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, v, w, p) while the `B`s are single-field spaces for 
buoyancy.
"""
function Spaces(mesh::Mesh, u_diri, v_diri, w_diri, b_diri)
    model = mesh.model

    # reference FE 
    reffe_p1 = ReferenceFE(lagrangian, Float64, 1; space=:P)
    reffe_b  = ReferenceFE(bubble, Float64)

    # test FESpaces
    u_diri_tags = collect(keys(u_diri))
    v_diri_tags = collect(keys(v_diri))
    w_diri_tags = collect(keys(w_diri))
    b_diri_tags = collect(keys(b_diri))
    @info "Building `Gridap.TestFESpace`s..."
    @time begin
    U_test  = TestFESpace(model, reffe_p1, conformity=:H1, dirichlet_tags=(length(u_diri_tags) > 0) ? u_diri_tags : Int[])
    UB_test = TestFESpace(model, reffe_b)
    V_test  = TestFESpace(model, reffe_p1, conformity=:H1, dirichlet_tags=(length(v_diri_tags) > 0) ? v_diri_tags : Int[])
    VB_test = TestFESpace(model, reffe_b)
    W_test  = TestFESpace(model, reffe_p1, conformity=:H1, dirichlet_tags=(length(w_diri_tags) > 0) ? w_diri_tags : Int[])
    WB_test = TestFESpace(model, reffe_b)
    P_test  = TestFESpace(model, reffe_p1, conformity=:H1, constraint=:zeromean)
    X_test  = MultiFieldFESpace([U_test, UB_test, V_test, VB_test, W_test, WB_test, P_test])
    B_test  = TestFESpace(model, reffe_p1, conformity=:H1, dirichlet_tags=(length(b_diri_tags) > 0) ? b_diri_tags : Int[])
    end

    # trial FESpaces with Dirichlet values
    u_diri_vals = collect(values(u_diri))
    v_diri_vals = collect(values(v_diri))
    w_diri_vals = collect(values(w_diri))
    b_diri_vals = collect(values(b_diri))
    @info "Building `Gridap.TrialFESpace`s..."
    @time begin
    if length(u_diri_vals) > 0
        U_trial = TrialFESpace(U_test, u_diri_vals)
    else
        U_trial = TrialFESpace(U_test)
    end
    UB_trial = TrialFESpace(UB_test)
    if length(v_diri_vals) > 0
        V_trial = TrialFESpace(V_test, v_diri_vals)
    else
        V_trial = TrialFESpace(V_test)
    end
    VB_trial = TrialFESpace(VB_test)
    if length(w_diri_vals) > 0
        W_trial = TrialFESpace(W_test, w_diri_vals)
    else
        W_trial = TrialFESpace(W_test)
    end
    WB_trial = TrialFESpace(WB_test)
    P_trial = TrialFESpace(P_test)
    X_trial = MultiFieldFESpace([U_trial, UB_trial, V_trial, VB_trial, W_trial, WB_trial, P_trial])
    if length(b_diri_vals) > 0
        B_trial = TrialFESpace(B_test, b_diri_vals)
    else
        B_trial = TrialFESpace(B_test)
    end
    end

    # a FEFunction that is b₀ on the dirichlet boundary and 0 elsewhere
    # (needed for assembling matrices)
    b_diri = interpolate(0, B_trial)

    return Spaces(X_trial, X_test, B_trial, B_test, b_diri)
end