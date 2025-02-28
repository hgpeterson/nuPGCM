"""
    X_trial, X_test, B_trial, B_test = setup_FESpaces(mesh; order=2)

Setup the trial and test FE spaces for the velocity, pressure, and buoyancy fields.
`mesh` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field FE spaces for (u, v, w, p) while the `B`s are single-field FE spaces 
for buoyancy.
"""
function setup_FESpaces(mesh; order=2)
    # reference FE 
    reffe_u = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_v = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_w = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_p = ReferenceFE(lagrangian, Float64, order-1; space=:P)
    reffe_b = ReferenceFE(lagrangian, Float64, order;   space=:P)

    # test FESpaces
    U_test = TestFESpace(mesh, reffe_u, conformity=:H1, dirichlet_tags=["bot"])
    V_test = TestFESpace(mesh, reffe_v, conformity=:H1, dirichlet_tags=["bot"])
    W_test = TestFESpace(mesh, reffe_w, conformity=:H1, dirichlet_tags=["bot", "sfc"])
    P_test = TestFESpace(mesh, reffe_p, conformity=:H1, constraint=:zeromean)
    B_test = TestFESpace(mesh, reffe_b, conformity=:H1, dirichlet_tags=["sfc"])
    X_test = MultiFieldFESpace([U_test, V_test, W_test, P_test])

    # trial FESpaces with Dirichlet values
    U_trial = TrialFESpace(U_test, [0])
    V_trial = TrialFESpace(V_test, [0])
    W_trial = TrialFESpace(W_test, [0, 0])
    P_trial = TrialFESpace(P_test)
    B_trial = TrialFESpace(B_test, [0])
    X_trial = MultiFieldFESpace([U_trial, V_trial, W_trial, P_trial])

    return X_trial, X_test, B_trial, B_test
end

"""
    spaces = unpack_spaces(X::MultiFieldFESpace)

Return the individual field spaces of the multi-field FE space `X`.

Example
===

```julia
julia> X, Y, B, D = setup_FESpaces(mesh)
(MultiFieldFESpace(), MultiFieldFESpace(), TrialFESpace(), UnconstrainedFESpace())

julia> U, V, W, P = unpack_spaces(X)
4-element Vector{Gridap.FESpaces.SingleFieldFESpace}:
 TrialFESpace()
 TrialFESpace()
 TrialFESpace()
 ZeroMeanFESpace()
```
"""
function unpack_spaces(X::MultiFieldFESpace)
    return [X.spaces[i] for i in eachindex(X.spaces)] 
end