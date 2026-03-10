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
    spaces = Spaces(mesh::Mesh, u_diri, v_diri, w_diri, b_diri; u_order=2, b_order=2)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.

`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, v, w, p) while the `B`s are single-field spaces for 
buoyancy.
"""
function Spaces(mesh::Mesh; u_diri_tags=Int[], u_diri_masks=Int[], u_diri_vals=nothing, 
                            b_diri_tags=Int[], b_diri_vals=nothing, 
                            u_order=2, b_order=2)
    model = mesh.model

    # reference FE 
    reffe_u = ReferenceFE(lagrangian, VectorValue{3, Float64}, u_order;   space=:P)
    reffe_p = ReferenceFE(lagrangian, Float64,                 u_order-1; space=:P)
    reffe_b = ReferenceFE(lagrangian, Float64,                 b_order;   space=:P)

    # test FESpaces
    @info "Building `Gridap.TestFESpace`s..."
    @time begin
    U_test = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=u_diri_tags, dirichlet_masks=u_diri_masks)
    P_test = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    X_test = MultiFieldFESpace([U_test, P_test])
    B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=b_diri_tags)
    end

    # trial FESpaces with Dirichlet values
    @info "Building `Gridap.TrialFESpace`s..."
    @time begin
    if u_diri_vals !== nothing
        U_trial = TrialFESpace(U_test, [VectorValue(v) for v in u_diri_vals])
    else
        U_trial = TrialFESpace(U_test)
    end
    P_trial = TrialFESpace(P_test)
    X_trial = MultiFieldFESpace([U_trial, P_trial])
    if b_diri_vals !== nothing
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