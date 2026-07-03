struct FEData{M<:Mesh, DUP, DB, CUP, CB}
    mesh::M
    dh_up::DUP    # combined DofHandler for (u, p)
    dh_b::DB      # DofHandler for buoyancy b used for evolution
    ch_up::CUP    # ConstraintHandler for (u, p)
    ch_b::CB      # ConstraintHandler for b
    u_dof_indices::Vector{Int}    # sorted global u DOF indices within dh_up
    p_dof_indices::Vector{Int}    # sorted global p DOF indices within dh_up
    nu::Int
    np::Int
    nb::Int
    u_order::Int
    b_order::Int
    p_up::Vector{Int}        # permutation for (u, p) system
    p_b::Vector{Int}         # permutation for b
    inv_p_up::Vector{Int}    # inverse permutations
    inv_p_b::Vector{Int}
end

function Base.summary(fe_data::FEData)
    t = typeof(fe_data)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, fe_data::FEData)
    println(io, summary(fe_data), ":")
    println(io, "├── mesh: ", summary(fe_data.mesh))
    println(io, "├── nu = ", fe_data.nu)
    println(io, "├── np = ", fe_data.np)
      print(io, "└── nb = ", fe_data.nb)
end

"""
    fe_data = FEData(mesh;
                     u_diri_tags, u_diri_masks, u_diri_vals,
                     b_diri_tags, b_diri_vals,
                     u_order, b_order)

Set up `DofHandler`s, `ConstraintHandler`s, and DOF permutations for the PG model.

A combined `DofHandler` for (u, p) is used for the inversion problem so that
Ferrite's `apply!` can correctly handle periodic BCs (which require entries between
image and mirror DOFs that cross cell boundaries). `u_dof_indices` and `p_dof_indices`
store the global DOF index sets within `dh_up` for splitting the combined solution.

Periodic BCs are applied automatically when `"channel_west"` and `"channel_east"`
facetsets are present in the grid. The periodic translation vector is inferred from
the east-wall x-coordinate.

DOF permutations are identity for now; Cuthill-McKee reordering will be added
once the mass matrix assembly is in place.
"""
function FEData(mesh::Mesh;
                u_diri_tags  = String[],
                u_diri_masks = nothing,   # e.g. [(true,true,true), (false,false,true)]
                u_diri_vals  = nothing,   # currently unused (always zero for u)
                b_diri_tags  = String[],
                b_diri_vals  = nothing,   # function b(x) or constant
                u_order = 2,
                b_order = 2)
    grid = mesh.grid

    ip_u = Lagrange{RefTetrahedron, u_order}()^3
    ip_p = Lagrange{RefTetrahedron, u_order - 1}()
    ip_b = Lagrange{RefTetrahedron, b_order}()

    @info "Building DofHandlers..."
    @time begin
    dh_up = DofHandler(grid); add!(dh_up, :u, ip_u); add!(dh_up, :p, ip_p); close!(dh_up)
    dh_b  = DofHandler(grid); add!(dh_b,  :b, ip_b); close!(dh_b)
    end

    # collect sorted u and p DOF index sets from the combined DofHandler
    u_set = Set{Int}()
    p_set = Set{Int}()
    for cc in CellIterator(dh_up)
        dofs = celldofs(cc)
        union!(u_set, dofs[dof_range(dh_up, :u)])
        union!(p_set, dofs[dof_range(dh_up, :p)])
    end
    u_dof_indices = sort!(collect(u_set))
    p_dof_indices = sort!(collect(p_set))
    nu = length(u_dof_indices)
    np = length(p_dof_indices)
    nb = ndofs(dh_b)

    # periodic facet pairs, auto-detected from facetset names
    is_periodic = haskey(grid.facetsets, "channel_west") &&
                  haskey(grid.facetsets, "channel_east")
    if is_periodic
        W = _channel_width(grid)
        pfacets = collect_periodic_facets(grid, "channel_west", "channel_east",
                                          x -> x - Vec{3}((W, 0.0, 0.0)))
        @info @sprintf("Periodic mesh: channel width W = %.4f, %d face pairs", W, length(pfacets))
    end

    @info "Building ConstraintHandlers..."
    @time begin

    ch_up = ConstraintHandler(dh_up)
    for (i, tag) in enumerate(u_diri_tags)
        components = _mask_to_components(u_diri_masks, i)
        if components === nothing
            add!(ch_up, Dirichlet(:u, grid.facetsets[tag], (x, t) -> zero(Vec{3, Float64})))
        else
            add!(ch_up, Dirichlet(:u, grid.facetsets[tag], (x, t) -> 0.0, components))
        end
    end
    if is_periodic
        add!(ch_up, PeriodicDirichlet(:u, pfacets))
        add!(ch_up, PeriodicDirichlet(:p, pfacets))
        # Mean pressure constraint: ∫p dΩ = 0. PeriodicDirichlet(:p, pfacets) makes the
        # "channel_west" (mirror) pressure DOFs dependent on "channel_east" (image) DOFs,
        # so the mean constraint must avoid using any "channel_west" DOF as its dependent
        # DOF or as a term in its affine combination -- otherwise close! raises "nested
        # affine constraints currently not supported".
        excluded = _dirichlet_dof_set(dh_up, :p, grid.facetsets["channel_west"])
        add!(ch_up, _mean_pressure_constraint(dh_up, u_order - 1; excluded))
    else
        add!(ch_up, _mean_pressure_constraint(dh_up, u_order - 1))
    end
    close!(ch_up)
    Ferrite.update!(ch_up, 0.0)

    ch_b = ConstraintHandler(dh_b)
    for (i, tag) in enumerate(b_diri_tags)
        f = _to_dirichlet_fn(b_diri_vals !== nothing ? b_diri_vals[i] : nothing)
        add!(ch_b, Dirichlet(:b, grid.facetsets[tag], f))
    end
    is_periodic && add!(ch_b, PeriodicDirichlet(:b, pfacets))
    close!(ch_b)
    Ferrite.update!(ch_b, 0.0)

    end

    # identity permutations (Cuthill-McKee reordering not yet implemented)
    N_up     = ndofs(dh_up)
    p_up     = collect(1:N_up)
    p_b      = collect(1:nb)
    inv_p_up = collect(1:N_up)
    inv_p_b  = collect(1:nb)

    return FEData(mesh, dh_up, dh_b, ch_up, ch_b,
                  u_dof_indices, p_dof_indices,
                  nu, np, nb, u_order, b_order,
                  p_up, p_b, inv_p_up, inv_p_b)
end

"""
    nu, np, nb = get_n_dofs(fe_data)
"""
get_n_dofs(fe_data::FEData) = fe_data.nu, fe_data.np, fe_data.nb

### helpers

function _channel_width(grid)
    return maximum(
        grid.nodes[n].x[1]
        for fi in grid.facetsets["channel_east"]
        for n in Ferrite.facets(grid.cells[fi[1]])[fi[2]]
    )
end

function _mask_to_components(masks, i)
    masks === nothing && return nothing
    mask = masks[i]
    comps = [j for j in 1:3 if mask[j]]
    length(comps) == 3 && return nothing   # all components: omit keyword
    return comps
end

"""
    _condense_rhs!(y, ch)

Prepare a physics RHS vector `y` for solving against a pre-condensed matrix.

For each affine constraint `y[constrained] = Σ c * y[free]` (for periodic BCs:
mirror = constrained, image = free): merges the constrained DOF's row into the
rows of its coefficient DOFs (`y[free] += c * y[constrained]`), then zeros all
constrained DOF rows (pure Dirichlet rows included).

This is the correct right-hand-side operation when the stiffness matrix has already
been condensed by `condense_system` (or Ferrite's `apply!(K, f_bc, ch)`) and the
solution will be corrected afterward by `apply!(x, ch)`. Unlike `apply!(y, ch)`, this
does not overwrite constrained rows with their recovered values (which would corrupt
the condensed system's equations for those rows).
"""
function _condense_rhs!(y::AbstractVector, ch::ConstraintHandler)
    for i in eachindex(ch.prescribed_dofs, ch.dofcoefficients)
        dofcoef = ch.dofcoefficients[i]
        dofcoef === nothing && continue
        cdof = ch.prescribed_dofs[i]
        for (fdof, c) in dofcoef
            y[fdof] += c * y[cdof]
        end
    end
    y[ch.prescribed_dofs] .= 0.0
    return y
end

"""
    A_cond, f_bc = condense_system(A, ch)

Condense the matrix `A` with the constraints in `ch`: return `CᵀAC + D`, where
`C` maps reduced to full DOFs (`x_full = C x_reduced + g`, `g` the constraint
inhomogeneities) and `D` places the mean |diagonal| on constrained rows, plus
the RHS correction `f_bc = Cᵀ(-A g)` (zero when all constraints are homogeneous).

This replaces Ferrite's `apply!(A, f_bc, ch)`, which mis-places the coupling
blocks between pairs of constrained DOFs in non-symmetric matrices: with
constraints `x[c1] = x[f1]` and `x[c2] = x[f2]`, it folds `A[c1, c2]` into the
transposed position `A[f2, f1]` instead of `A[f1, f2]` (Ferrite ≤ 1.4,
`_condense!`). With periodic BCs this corrupts the divergence and Coriolis
blocks across the periodic seam; symmetric matrices (mass, diffusion) are
unaffected.

At solve time, prepare the RHS with `_condense_rhs!(y, ch)` and recover the
constrained solution DOFs with `apply!(x, ch)`, as with Ferrite's `apply!`.
"""
function condense_system(A::SparseMatrixCSC, ch::ConstraintHandler)
    N = size(A, 1)

    # RHS correction from inhomogeneous constraint values
    g = zeros(N)
    g[ch.prescribed_dofs] .= ch.inhomogeneities
    f_bc = -(A * g)
    _condense_rhs!(f_bc, ch)

    # constraint map C: identity on free DOFs, coefficient entries on constrained rows
    rows = collect(1:N); cols = collect(1:N); vals = ones(N)
    for (i, cdof) in enumerate(ch.prescribed_dofs)
        vals[cdof] = 0.0   # no identity row for constrained DOFs
        dofcoef = ch.dofcoefficients[i]
        dofcoef === nothing && continue
        for (fdof, c) in dofcoef
            push!(rows, cdof); push!(cols, fdof); push!(vals, c)
        end
    end
    C = sparse(rows, cols, vals, N, N)

    # mean |diagonal| keeps constrained rows at a scale comparable to A
    md = sum(abs, diag(A)) / N
    D = sparse(ch.prescribed_dofs, ch.prescribed_dofs,
               fill(md, length(ch.prescribed_dofs)), N, N)

    return C' * A * C + D, f_bc
end

function _to_dirichlet_fn(val)
    val === nothing  && return (x, t) -> 0.0
    val isa Function && return (x, t) -> val(x)
    return (x, t) -> val
end

"""
    _dirichlet_dof_set(dh, field, facetset) -> Set{Int}

Return the set of global DOF indices for `field` on `facetset`, using a throwaway
`ConstraintHandler` to harvest `Dirichlet`'s own facet-to-dof mapping.
"""
function _dirichlet_dof_set(dh, field::Symbol, facetset)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(field, facetset, (x, t) -> 0.0))
    close!(ch)
    return Set(ch.prescribed_dofs)
end

"""
    _mean_pressure_constraint(dh_up, p_order; excluded = nothing) -> AffineConstraint

Build an `AffineConstraint` that enforces ∫p dΩ = 0 by expressing the
pressure DOF with the largest volume-integral weight as a linear combination
of all other pressure DOFs.

Assembles the 1×N_p constraint row `C[1, i] = ∫φ_p_i dΩ`, then picks DOF k
with |C[1,k]| largest as the dependent DOF:
    p[k] = -∑_{i≠k} (C[1,i]/C[1,k]) p[i]

If `excluded` is given, those DOFs are dropped from consideration entirely (neither
chosen as the dependent DOF `k` nor included as an independent term), so the result
can coexist with other affine constraints (e.g. periodic) already touching them.
"""
function _mean_pressure_constraint(dh_up, p_order; excluded = nothing)
    ip_p    = Lagrange{RefTetrahedron, p_order}()
    ip_geo  = Lagrange{RefTetrahedron, 1}()
    qr      = QuadratureRule{RefTetrahedron}(2 * p_order + 1)
    cv_p    = CellValues(qr, ip_p, ip_geo)
    n_p     = getnbasefunctions(cv_p)
    range_p = dof_range(dh_up, :p)
    Ce      = zeros(1, n_p)

    assembler = Ferrite.COOAssembler()
    for cc in CellIterator(dh_up)
        reinit!(cv_p, cc)
        fill!(Ce, 0.0)
        for q in 1:getnquadpoints(cv_p)
            dΩ = getdetJdV(cv_p, q)
            for i in 1:n_p
                Ce[1, i] += shape_value(cv_p, q, i) * dΩ
            end
        end
        assemble!(assembler, [1], collect(celldofs(cc)[range_p]), Ce)
    end

    C, _ = finish_assemble(assembler)
    _, J, V = findnz(C)
    if excluded !== nothing
        keep = [i for i in eachindex(J) if J[i] ∉ excluded]
        J, V = J[keep], V[keep]
    end
    _, idx  = findmax(abs2, V)
    cdof    = J[idx]
    V     ./= V[idx]
    return AffineConstraint(
        cdof,
        Pair{Int,Float64}[J[i] => -V[i] for i in eachindex(J) if J[i] != cdof],
        0.0,
    )
end
