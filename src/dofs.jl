struct FEData{M<:Mesh, DU, DP, DB, CU, CP, CB}
    mesh::M
    dh_u::DU          # DofHandler for velocity u
    dh_p::DP          # DofHandler for pressure p
    dh_b::DB          # DofHandler for buoyancy b
    ch_u::CU          # ConstraintHandler for u (Dirichlet + periodic)
    ch_p::CP          # ConstraintHandler for p (periodic only)
    ch_b::CB          # ConstraintHandler for b (Dirichlet + periodic)
    nu::Int
    np::Int
    nb::Int
    u_order::Int
    b_order::Int
    p_up::Vector{Int}      # permutation for combined [u; p] system
    p_b::Vector{Int}       # permutation for b
    inv_p_up::Vector{Int}  # inverse permutations
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

Periodic BCs are applied automatically when `"channel_west"` and `"channel_east"`
facetsets are present in the grid (channel-basin geometry). The periodic translation
vector is inferred from the east-wall x-coordinate.

DOF permutations are identity for now; Cuthill-McKee reordering will be added in
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
    dh_u = DofHandler(grid); add!(dh_u, :u, ip_u); close!(dh_u)
    dh_p = DofHandler(grid); add!(dh_p, :p, ip_p); close!(dh_p)
    dh_b = DofHandler(grid); add!(dh_b, :b, ip_b); close!(dh_b)
    end

    nu = ndofs(dh_u)
    np = ndofs(dh_p)
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

    ch_u = ConstraintHandler(dh_u)
    for (i, tag) in enumerate(u_diri_tags)
        components = _mask_to_components(u_diri_masks, i)
        if components === nothing
            add!(ch_u, Dirichlet(:u, grid.facetsets[tag], (x, t) -> zero(Vec{3, Float64})))
        else
            add!(ch_u, Dirichlet(:u, grid.facetsets[tag], (x, t) -> 0.0, components))
        end
    end
    is_periodic && add!(ch_u, PeriodicDirichlet(:u, pfacets))
    close!(ch_u)
    update!(ch_u, 0.0)

    ch_p = ConstraintHandler(dh_p)
    is_periodic && add!(ch_p, PeriodicDirichlet(:p, pfacets))
    close!(ch_p)
    update!(ch_p, 0.0)

    ch_b = ConstraintHandler(dh_b)
    for (i, tag) in enumerate(b_diri_tags)
        f = _to_dirichlet_fn(b_diri_vals !== nothing ? b_diri_vals[i] : nothing)
        add!(ch_b, Dirichlet(:b, grid.facetsets[tag], f))
    end
    is_periodic && add!(ch_b, PeriodicDirichlet(:b, pfacets))
    close!(ch_b)
    update!(ch_b, 0.0)

    end

    # identity permutations (Cuthill-McKee reordering not yet implemented)
    p_up     = collect(1:nu + np)
    p_b      = collect(1:nb)
    inv_p_up = collect(1:nu + np)
    inv_p_b  = collect(1:nb)

    return FEData(mesh, dh_u, dh_p, dh_b, ch_u, ch_p, ch_b,
                  nu, np, nb, u_order, b_order, p_up, p_b, inv_p_up, inv_p_b)
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
