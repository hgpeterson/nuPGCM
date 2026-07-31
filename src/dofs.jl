struct FEData{M<:Mesh, DUP, DB, CUP, CB}
    mesh::M
    dh_up::DUP    # combined DofHandler for (u, p)
    dh_b::DB      # DofHandler for buoyancy b used for evolution
    ch_up::CUP    # ConstraintHandler for (u, p)
    ch_b::CB      # ConstraintHandler for b
    u_dof_indices::Vector{Int}    # sorted global u DOF indices within dh_up
    p_dof_indices::Vector{Int}    # sorted global p DOF indices within dh_up
    free_dofs::Vector{Int}        # sorted unconstrained DOF indices within dh_up
    nu::Int
    np::Int
    nb::Int
    u_order::Int
    b_order::Int
    cache::AssemblyCache     # mesh-constant data for per-timestep re-assembly
    C_up::SparseMatrixCSC{Float64, Int}   # N_up × N_free constraint map (see condense_system)
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
                     u_diri_tags, u_diri_masks,
                     b_diri_tags, b_diri_vals,
                     u_order, b_order, pressure_gauge)

Set up `DofHandler`s and `ConstraintHandler`s for the PG model.

A combined `DofHandler` for (u, p) is used for the inversion problem so that
periodic BCs (which couple image and mirror DOFs across cell boundaries) can be
condensed consistently. `u_dof_indices` and `p_dof_indices` store the global DOF
index sets within `dh_up` for splitting the combined solution; `free_dofs` stores
the unconstrained DOFs, which are the ones actually solved for (see
[`condense_system`](@ref)).

Periodic BCs are applied automatically when `"channel_west"` and `"channel_east"`
facetsets are present in the grid. The periodic translation vector is inferred from
the east-wall x-coordinate. Dirichlet values for `u` are always zero.

`pressure_gauge` defaults to `:pin`. `:mean` is correct but expensive: its
constraint expresses one pressure DOF as a dense combination of *all* the others,
which makes Ferrite's `allocate_matrix(dh, ch)` fire its "double-distribute"
branch and insert ~`np²` entries (88.7M vs 17.0M nonzeros, 3.4 GiB vs 780 MiB of
GPU memory at h=4e-2; the block grows as `np²`). `:pin` fixes a single DOF instead,
leaving pressure determined up to a constant — subtract the mean at output time if
a zero-mean field is wanted. See `scratch/diagnose_gauge_nnz.jl`.
"""
function FEData(mesh::Mesh;
                u_diri_tags  = String[],
                u_diri_masks = nothing,   # e.g. [(true,true,true), (false,false,true)]
                b_diri_tags  = String[],
                b_diri_vals  = nothing,   # function b(x) or constant
                u_order = 2,
                b_order = 2,
                pressure_gauge = :pin)    # :pin (one DOF = 0), :mean (∫p dΩ = 0), :none
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
    end
    if pressure_gauge != :none
        # Pressure gauge. With periodic BCs, PeriodicDirichlet(:p, pfacets) makes the
        # "channel_west" (mirror) pressure DOFs dependent on "channel_east" (image) DOFs,
        # so the gauge constraint must avoid using any "channel_west" DOF as its dependent
        # DOF or as a term in its affine combination -- otherwise close! raises "nested
        # affine constraints currently not supported".
        excluded = is_periodic ?
            _dirichlet_dof_set(dh_up, :p, grid.facetsets["channel_west"]) : nothing
        gauge = _mean_pressure_constraint(dh_up, u_order - 1; excluded)
        if pressure_gauge == :pin
            # pin the same DOF the mean constraint would eliminate: p[cdof] = 0.
            # Sparse and iterative-solver friendly; pressure is then determined up to
            # the gauge instead of having zero mean.
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
        f = _to_dirichlet_fn(b_diri_vals !== nothing ? b_diri_vals[i] : nothing)
        add!(ch_b, Dirichlet(:b, grid.facetsets[tag], f))
    end
    is_periodic && add!(ch_b, PeriodicDirichlet(:b, pfacets))
    close!(ch_b)
    Ferrite.update!(ch_b, 0.0)

    end

    N_up      = ndofs(dh_up)
    free_dofs = setdiff(1:N_up, ch_up.prescribed_dofs)
    @info @sprintf("Inversion DOFs: %d total, %d prescribed, %d solved for",
                   N_up, length(ch_up.prescribed_dofs), length(free_dofs))

    @info "Building assembly cache..."
    @time begin
    cache = AssemblyCache(dh_up, dh_b, u_order, b_order)
    C_up  = _constraint_matrix(ch_up, N_up, free_dofs)
    end

    return FEData(mesh, dh_up, dh_b, ch_up, ch_b,
                  u_dof_indices, p_dof_indices, free_dofs,
                  nu, np, nb, u_order, b_order,
                  cache, C_up)
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
    A_red, f_bc = condense_system(A, ch, C)

Reduce the `N × N` matrix `A` to the `N_free × N_free` system actually solved,
using the constraint map `C` (`N × N_free`, from [`_constraint_matrix`](@ref)):

    A_red = CᵀAC,   f_bc = Cᵀ(-A g)

where `x_full = C x_red + g` and `g` carries the constraint inhomogeneities
(`f_bc` is zero when all constraints are homogeneous, as for this model's
velocity BCs).

Prescribed DOFs are *eliminated*, not retained: `C` has no column for them, so
they contribute no row or column to `A_red`. This is what Gridap does at
assembly time on `main` (`assemble_matrix` over a `TrialFESpace` emits the free
block directly, with the Dirichlet contribution as a separate RHS lift); here
the same reduced system is reached by projection after assembling the full one.

This also replaces Ferrite's `apply!(A, f_bc, ch)`, which mis-places the coupling
blocks between pairs of constrained DOFs in non-symmetric matrices: with
constraints `x[c1] = x[f1]` and `x[c2] = x[f2]`, it folds `A[c1, c2]` into the
transposed position `A[f2, f1]` instead of `A[f1, f2]` (Ferrite ≤ 1.4,
`_condense!`). With periodic BCs this corrupts the divergence and Coriolis
blocks across the periodic seam; symmetric matrices (mass, diffusion) are
unaffected.

At solve time, map the RHS with [`condense_rhs`](@ref), solve for `x_red`, then
scatter back into a full-length vector at `fe_data.free_dofs` and call
`apply!(x_full, ch)` to recover the constrained DOFs.
"""
function condense_system(A::SparseMatrixCSC, ch::ConstraintHandler, C::SparseMatrixCSC)
    g = zeros(size(A, 1))
    g[ch.prescribed_dofs] .= ch.inhomogeneities
    return C' * A * C, C' * (-(A * g))
end

"""
    y_red = condense_rhs(y, C)

Map a full-length physics RHS onto the reduced system: `y_red = Cᵀ y`.

Equivalent to [`_condense_rhs!`](@ref) followed by keeping only the free rows —
each constrained DOF's row is merged into the rows of its coefficient DOFs, and
the constrained rows themselves are dropped rather than zeroed.
"""
condense_rhs(y::AbstractVector, C::SparseMatrixCSC) = C' * y

"""
    C = _constraint_matrix(ch, N, free_dofs)

Build the `N × N_free` sparse constraint map `C` such that `x_full = C x_red + g`:
column `j` corresponds to `free_dofs[j]`, carrying a 1 on its own row plus the
coefficient of that DOF in every constrained DOF's affine combination.

Coefficient entries that reference a DOF which is itself prescribed (a
"junction" DOF, e.g. a periodic mirror whose image lies on a Dirichlet
boundary) are dropped: their value is already folded into `ch.inhomogeneities`
by Ferrite's `update!` and hence carried by `g` in [`condense_system`](@ref).
Keeping such an entry corrupts the condensed system at exactly those DOFs.
"""
function _constraint_matrix(ch::ConstraintHandler, N::Int, free_dofs::Vector{Int})
    # reduced index of each DOF (0 for prescribed DOFs, which have no column)
    red = zeros(Int, N)
    for (j, i) in enumerate(free_dofs)
        red[i] = j
    end

    rows = copy(free_dofs)
    cols = collect(1:length(free_dofs))
    vals = ones(length(free_dofs))
    for (i, cdof) in enumerate(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        dofcoef === nothing && continue
        for (fdof, c) in dofcoef
            haskey(ch.dofmapping, fdof) && continue   # prescribed: value lives in g
            push!(rows, cdof); push!(cols, red[fdof]); push!(vals, c)
        end
    end
    return sparse(rows, cols, vals, N, length(free_dofs))
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
