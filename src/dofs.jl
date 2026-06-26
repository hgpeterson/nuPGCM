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
    else
        # Mean pressure constraint: ∫p dΩ = 0. Skipped for periodic meshes because
        # PeriodicDirichlet already creates AffineConstraints on pressure image DOFs,
        # and Ferrite does not support nested affine constraints.
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

function _to_dirichlet_fn(val)
    val === nothing  && return (x, t) -> 0.0
    val isa Function && return (x, t) -> val(x)
    return (x, t) -> val
end

"""
    _mean_pressure_constraint(dh_up, p_order) -> AffineConstraint

Build an `AffineConstraint` that enforces ∫p dΩ = 0 by expressing the
pressure DOF with the largest volume-integral weight as a linear combination
of all other pressure DOFs.

Assembles the 1×N_p constraint row `C[1, i] = ∫φ_p_i dΩ`, then picks DOF k
with |C[1,k]| largest as the dependent DOF:
    p[k] = -∑_{i≠k} (C[1,i]/C[1,k]) p[i]
"""
function _mean_pressure_constraint(dh_up, p_order)
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
    _, idx  = findmax(abs2, V)
    cdof    = J[idx]
    V     ./= V[idx]
    return AffineConstraint(
        cdof,
        Pair{Int,Float64}[J[i] => -V[i] for i in eachindex(J) if J[i] != cdof],
        0.0,
    )
end
