"""
    X, Y, B, D = setup_FESpaces(model; order=2)

Setup the trial and test FE spaces for the velocity, pressure, and buoyancy fields.
`X` and `Y` are multi-field FE spaces for (u, v, w, p).
"""
function setup_FESpaces(model; order=2)
    # reference FE 
    reffe_ux = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_uy = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_uz = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_p  = ReferenceFE(lagrangian, Float64, order-1; space=:P)
    reffe_b  = ReferenceFE(lagrangian, Float64, order;   space=:P)

    # test FESpaces
    Vx = TestFESpace(model, reffe_ux, conformity=:H1, dirichlet_tags=["bot"])
    Vy = TestFESpace(model, reffe_uy, conformity=:H1, dirichlet_tags=["bot"])
    Vz = TestFESpace(model, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
    Q  = TestFESpace(model, reffe_p,  conformity=:H1, constraint=:zeromean)
    D  = TestFESpace(model, reffe_b,  conformity=:H1, dirichlet_tags=["sfc"])
    Y = MultiFieldFESpace([Vx, Vy, Vz, Q])

    # trial FESpaces with Dirichlet values
    Ux = TrialFESpace(Vx, [0])
    Uy = TrialFESpace(Vy, [0])
    Uz = TrialFESpace(Vz, [0, 0])
    P  = TrialFESpace(Q)
    B  = TrialFESpace(D, [0])
    X  = MultiFieldFESpace([Ux, Uy, Uz, P])

    return X, Y, B, D
end

"""
    spaces = unpack_spaces(X::MultiFieldFESpace)

Return the individual field spaces of the multi-field FE space `X`.

Example
===

```julia
julia> X, Y, B, D = setup_FESpaces(model)
(MultiFieldFESpace(), MultiFieldFESpace(), TrialFESpace(), UnconstrainedFESpace())

julia> Ux, Uy, Uz, P = unpack_spaces(X)
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