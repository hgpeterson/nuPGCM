# Bubble function conformity workaround (https://github.com/gridap/Gridap.jl/issues/1195)
ReferenceFEs.Conformity(::GenericRefFE{Bubble}, ::Symbol) = L2Conformity()

struct Spaces{X1, X2, B1, B2, BD}
    X::X1  # trial space [U, UB, P]
    Y::X2  # test space [V, VB, Q]
    B::B1  # trial space for buoyancy
    D::B2  # test space for buoyancy
    b_diri::BD   # FEFunction that is b₀ on the dirichlet boundary and 0 elsewhere
end

function Base.getproperty(spaces::Spaces, sym::Symbol) 
    if sym === :U
        return spaces.X[1]
    elseif sym === :UB
        return spaces.X[2]
    elseif sym === :P
        return spaces.X[3]
    elseif sym === :V
        return spaces.Y[1]
    elseif sym === :VB
        return spaces.Y[2]
    elseif sym === :Q
        return spaces.Y[3]
    else
        return getfield(spaces, sym)
    end
end

function Base.summary(spaces::Spaces)
    t = typeof(spaces)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, spaces::Spaces)
    println(io, summary(spaces), ":")
    println(io, "├── X: ", summary(spaces.X))
    println(io, "├── Y: ", summary(spaces.Y))
    println(io, "├── B: ", summary(spaces.B))
    println(io, "├── D: ", summary(spaces.D))
      print(io, "└── b_diri: ", summary(spaces.b_diri))
end

"""
    spaces = Spaces(mesh::Mesh, u_diri, v_diri, w_diri, b_diri)

Setup the trial and test spaces for the velocity, pressure, and buoyancy fields.

`model` is assumed to be an `UnstructuredDiscreteModel` from Gridap. The `X`s are 
multi-field spaces for (u, v, w, p) while the `B`s are single-field spaces for 
buoyancy.
"""
function Spaces(mesh::Mesh)
    model = mesh.model

    # reference FE 
    reffe_u = ReferenceFE(lagrangian, VectorValue{3, Float64}, 1; space=:P)
    reffe_b = ReferenceFE(bubble, VectorValue{3, Float64})
    reffe_p = ReferenceFE(lagrangian, Float64, 1; space=:P)

    # test FESpaces
    @info "Building `Gridap.TestFESpace`s..."
    @time begin
    V = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["bottom", "coastline", "surface"],
                    dirichlet_masks=[(true, true, true), (true, true, true), (false, false, true)])
    VB = TestFESpace(model, reffe_b, conformity=:L2)  # https://github.com/gridap/Gridap.jl/issues/1195
    Q = TestFESpace(model, reffe_p, conformity=:H1, constraint=:zeromean)
    Y = MultiFieldFESpace([V, VB, Q])
    D = TestFESpace(model, reffe_p, conformity=:H1, dirichlet_tags=["coastline", "surface"])
    end

    # trial FESpaces with Dirichlet values
    @info "Building `Gridap.TrialFESpace`s..."
    @time begin
    f = VectorValue{3, Float64}(0, 0, 0)
    U = TrialFESpace(V, [f, f, f])
    UB = TrialFESpace(VB)
    P = TrialFESpace(Q)
    X = MultiFieldFESpace([U, UB, P])
    B = TrialFESpace(D, [0, 0])
    end

    # a FEFunction that is b₀ on the dirichlet boundary and 0 elsewhere
    # (needed for assembling matrices)
    b_diri = interpolate(0, B)

    return Spaces(X, Y, B, D, b_diri)
end