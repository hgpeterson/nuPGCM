struct Spaces{X, Y, B, D, I}
    X::X
    Y::Y
    B::B
    D::D
    nx::I
    ny::I
    nz::I
    nu::I
    np::I
    nb::I
    N::I
end

function Spaces(geo::Geometry)
    # set up finite element spaces for problem
    X, Y, B, D = setup_spaces(geo.mesh)
    Ux, Uy, Uz, P = unpack_spaces(X)

    # number of degrees of freedom
    nx = Ux.space.nfree
    ny = Uy.space.nfree
    nz = Uz.space.nfree
    nu = nx + ny + nz
    np = P.space.space.nfree
    nb = B.space.nfree
    N = nu + np - 1
    @printf("\nN = %d (%d + %d) ∼ 10^%d DOF\n", N, nu, np-1, floor(log10(N)))

    return Spaces(X, Y, B, D, nx, ny, nz, nu, np, nb, N)
end

"""
    X, Y, B, D = setup_spaces(mesh; order=2)

Setup the trial and test FE spaces for the velocity, pressure, and buoyancy fields.
`X` and `Y` are multi-field FE spaces for (u, v, w, p).
"""
function setup_FESpaces(mesh::Gridap.Geometry.UnstructuredGrid; order=2)
    # reference FE 
    reffe_ux = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_uy = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_uz = ReferenceFE(lagrangian, Float64, order;   space=:P)
    reffe_p  = ReferenceFE(lagrangian, Float64, order-1; space=:P)
    reffe_b  = ReferenceFE(lagrangian, Float64, order;   space=:P)

    # test FESpaces
    Vx = TestFESpace(mesh, reffe_ux, conformity=:H1, dirichlet_tags=["bot"])
    Vy = TestFESpace(mesh, reffe_uy, conformity=:H1, dirichlet_tags=["bot"])
    Vz = TestFESpace(mesh, reffe_uz, conformity=:H1, dirichlet_tags=["bot", "sfc"])
    Q  = TestFESpace(mesh, reffe_p,  conformity=:H1, constraint=:zeromean)
    D  = TestFESpace(mesh, reffe_b,  conformity=:H1, dirichlet_tags=["sfc"])
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