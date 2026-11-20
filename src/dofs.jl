struct FEData{M<:Mesh, DUP, DB, CUP, CB, AC}
    mesh::M
    dh_up::DUP    # combined DofHandler for (u, p)
    dh_b::DB      # DofHandler for buoyancy b
    ch_up::CUP    # ConstraintHandler for (u, p)
    ch_b::CB      # ConstraintHandler for b
    u_dof_indices::Vector{Int}    # sorted global u DOF indices within dh_up
    p_dof_indices::Vector{Int}    # sorted global p DOF indices within dh_up
    free_dofs::Vector{Int}        # unconstrained DOFs of dh_up: velocity first, then pressure
    free_u::UnitRange{Int}        # positions of the velocity DOFs within free_dofs
    free_p::UnitRange{Int}        # positions of the pressure DOFs within free_dofs
    nu::Int
    np::Int
    nb::Int
    u_order::Int
    b_order::Int
    cache::AC
    C_up::SparseMatrixCSC{Float64, Int}   # constraint map, x_full = C_up * x_free
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
    fe_data = FEData(mesh; u_diri_tags, u_diri_masks, b_diri_tags, b_diri_vals,
                     u_order, b_order, pressure_gauge)

Set up the `DofHandler`s and `ConstraintHandler`s of the PG model: a combined one
for `(u, p)` and one for `b`.

- `u_diri_tags`: facetsets where `u = 0`. `u_diri_masks[i]` selects the components
  constrained on `u_diri_tags[i]`, e.g. `(false, false, true)` for `w = 0` only.
- `b_diri_tags`, `b_diri_vals`: facetsets where `b` is prescribed, and its value on
  each (a function of `x` or a constant).
- `pressure_gauge`: `:pin` sets one pressure DOF to zero; `:mean` enforces
  `∫p dΩ = 0`, which couples one DOF to all the others and adds ~`np²` matrix
  entries; `:none` leaves the constant pressure mode in the system.

`u`, `p`, and `b` are periodic across `mesh.periodic_facets`.
"""
function FEData(mesh::Mesh;
                u_diri_tags  = String[],
                u_diri_masks = nothing,
                b_diri_tags  = String[],
                b_diri_vals  = nothing,
                u_order = 2,
                b_order = 2,
                pressure_gauge = :pin)
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

    pfacets  = mesh.periodic_facets
    periodic = !isempty(pfacets)

    @info "Building ConstraintHandlers..."
    @time begin

    ch_up = ConstraintHandler(dh_up)
    for (i, tag) in enumerate(u_diri_tags)
        components = mask_to_components(u_diri_masks, i)
        if components === nothing
            add!(ch_up, Dirichlet(:u, grid.facetsets[tag], (x, t) -> zero(Vec{3, Float64})))
        else
            add!(ch_up, Dirichlet(:u, grid.facetsets[tag], (x, t) -> 0.0, components))
        end
    end
    if periodic
        add!(ch_up, PeriodicDirichlet(:u, pfacets))
        add!(ch_up, PeriodicDirichlet(:p, pfacets))
    end
    if pressure_gauge != :none
        # the gauge must avoid the pressure DOFs on the mirror facets, which the
        # periodic constraint already eliminates
        excluded = periodic ?
            dirichlet_dof_set(dh_up, :p, Set(pair.mirror for pair in pfacets)) : nothing
        gauge = mean_pressure_constraint(dh_up, u_order - 1; excluded)
        if pressure_gauge == :pin
            gauge = AffineConstraint(gauge.constrained_dof, Pair{Int, Float64}[], 0.0)
        elseif pressure_gauge != :mean
            throw(ArgumentError("pressure_gauge must be :mean, :pin, or :none"))
        end
        add!(ch_up, gauge)
    end
    close!(ch_up)
    Ferrite.update!(ch_up, 0.0)

    ch_b = ConstraintHandler(dh_b)
    for (i, tag) in enumerate(b_diri_tags)
        f = to_dirichlet_fn(b_diri_vals !== nothing ? b_diri_vals[i] : nothing)
        add!(ch_b, Dirichlet(:b, grid.facetsets[tag], f))
    end
    periodic && add!(ch_b, PeriodicDirichlet(:b, pfacets))
    close!(ch_b)
    Ferrite.update!(ch_b, 0.0)

    end

    N_up    = ndofs(dh_up)
    is_free = trues(N_up)
    is_free[ch_up.prescribed_dofs] .= false
    free_u_dofs = filter(i -> is_free[i], u_dof_indices)
    free_p_dofs = filter(i -> is_free[i], p_dof_indices)
    free_dofs   = vcat(free_u_dofs, free_p_dofs)
    free_u      = 1:length(free_u_dofs)
    free_p      = length(free_u_dofs) .+ (1:length(free_p_dofs))
    @info @sprintf("Inversion DOFs: %d total, %d prescribed, %d solved for (%d u, %d p)",
                   N_up, length(ch_up.prescribed_dofs), length(free_dofs),
                   length(free_u), length(free_p))

    @info "Building assembly cache..."
    @time begin
    cache = AssemblyCache(dh_up, dh_b, u_dof_indices, p_dof_indices, u_order, b_order)
    C_up  = constraint_matrix(ch_up, N_up, free_dofs)
    end

    return FEData(mesh, dh_up, dh_b, ch_up, ch_b,
                  u_dof_indices, p_dof_indices, free_dofs, free_u, free_p,
                  nu, np, nb, u_order, b_order,
                  cache, C_up)
end

"""
    nu, np, nb = get_n_dofs(fe_data)
"""
get_n_dofs(fe_data::FEData) = fe_data.nu, fe_data.np, fe_data.nb

### helpers

function mask_to_components(masks, i)
    masks === nothing && return nothing
    mask = masks[i]
    comps = [j for j in 1:3 if mask[j]]
    length(comps) == 3 && return nothing   # all components: omit keyword
    return comps
end

"""
    C = constraint_matrix(ch, N, free_dofs)

Sparse `N × N_free` map from the free DOFs to all `N` DOFs of a handler with
homogeneous constraints, `x_full = C x_free`. Column `j` has a 1 on row
`free_dofs[j]`, plus the coefficient of that DOF on the row of every constrained
DOF that depends on it.
"""
function constraint_matrix(ch::ConstraintHandler, N::Int, free_dofs::Vector{Int})
    column = zeros(Int, N)   # column of each free DOF
    for (j, i) in enumerate(free_dofs)
        column[i] = j
    end

    rows = copy(free_dofs)
    cols = collect(1:length(free_dofs))
    vals = ones(length(free_dofs))
    for (i, cdof) in enumerate(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        dofcoef === nothing && continue
        for (fdof, c) in dofcoef
            haskey(ch.dofmapping, fdof) && continue   # itself prescribed, to zero
            push!(rows, cdof); push!(cols, column[fdof]); push!(vals, c)
        end
    end
    return sparse(rows, cols, vals, N, length(free_dofs))
end

"""
    S = select_rows(A, I)

`A[I, :]` for a sparse `A`, through the columns of its transpose, in time linear in
`nnz(A)` and `length(I)`.
"""
select_rows(A::SparseMatrixCSC, I) = permutedims(permutedims(A)[:, I])

"""
    free = free_index(C)

For a constraint map `C` whose rows have at most one entry, equal to 1: the free DOF
that each DOF takes the value of, or 0 if it is eliminated.
"""
function free_index(C::SparseMatrixCSC)
    rows, columns, _ = findnz(C)
    free = zeros(Int, size(C, 1))
    free[rows] .= columns
    return free
end

function to_dirichlet_fn(val)
    val === nothing  && return (x, t) -> 0.0
    val isa Function && return (x, t) -> val(x)
    return (x, t) -> val
end

"""
    dofs = dirichlet_dof_set(dh, field, facetset)

Set of global DOF indices of `field` on `facetset`.
"""
function dirichlet_dof_set(dh, field::Symbol, facetset)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(field, facetset, (x, t) -> 0.0))
    close!(ch)
    return Set(ch.prescribed_dofs)
end

"""
    constraint = mean_pressure_constraint(dh_up, p_order; excluded = nothing)

`AffineConstraint` enforcing `∫p dΩ = 0`, written as the pressure DOF with the
largest weight `∫φᵢ dΩ` in terms of all the others. DOFs in `excluded` are left out.
"""
function mean_pressure_constraint(dh_up, p_order; excluded = nothing)
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
