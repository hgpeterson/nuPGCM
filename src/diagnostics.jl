####
#### Zonal (x) integration of a 3D field onto a structured (y, z) grid
####

"""
    lims = _grid_bbox(grid, dim)

Min/max node coordinate of the mesh along `dim` (1 = x, 2 = y, 3 = z).
"""
function _grid_bbox(grid, dim::Int)
    lo, hi = Inf, -Inf
    for node in grid.nodes
        c  = node.x[dim]
        lo = min(lo, c)
        hi = max(hi, c)
    end
    return (lo, hi)
end

"""
    x_in, x_out = _x_line_interval(c0, c1)

Interval of `x` over which the line `{(x, y₀, z₀)}` lies inside a tetrahedron,
given its reference coordinates along the line, `ξ(x) = c0 + c1 x`. Inside means
all four barycentric coordinates `ξ₁, ξ₂, ξ₃, 1 - ξ₁ - ξ₂ - ξ₃` are nonnegative;
each is affine in `x`, so each bounds `x` from one side. Empty when `x_out ≤ x_in`.
"""
@inline function _x_line_interval(c0::Vec{3}, c1::Vec{3})
    x_in, x_out = -Inf, Inf
    for (a, b) in ((c0[1], c1[1]), (c0[2], c1[2]), (c0[3], c1[3]),
                   (1 - c0[1] - c0[2] - c0[3], -c1[1] - c1[2] - c1[3]))
        if b > 0
            x_in  = max(x_in, -a/b)
        elseif b < 0
            x_out = min(x_out, -a/b)
        elseif a < 0             # parallel to this face and outside it
            return (Inf, -Inf)
        end
    end
    return (x_in, x_out)
end

"""
    res = zonal_integral(model, field; kwargs...)
    res = zonal_integral(fe_data, state, field; kwargs...)

Zonally (x-) integrate a solution field onto a structured `(y, z)` grid:

    Q(y, z) = ∫ q(x, y, z) dx

over the wet part of each x-line. `field` is one of `:u`, `:v`, `:w` (velocity
components), `:p`, or `:b`.

Returns a NamedTuple `(; y, z, Q, width, y_lims, z_lims)`:
- `y`, `z` — sample coordinates, lengths `ny` and `nz`
- `Q`      — the zonal integral, an `ny × nz` matrix
- `width`  — `∫ dx` over the same wet x-line, so `Q ./ width` is the zonal *mean*
             and `width == 0` marks dry points (under topography, or outside the
             domain). For a flat-bottomed box it must equal the channel width to
             roundoff, which makes it a direct check of the line clipping.
- `y_lims`, `z_lims` — the requested extent (the outer samples sit just inside it)

## Method

The integral is computed exactly (to roundoff) at each sample point, from the
full P2 fields, not from the P1 corner values a `VTKGridFile` would write. The
geometry is affine, so along the line `{(x, y₀, z₀)}` a tetrahedron's reference
coordinates are affine in `x`: the line crosses it over one interval, found from
where its barycentric coordinates change sign (see [`_x_line_interval`](@ref)).
On that interval a P2 field is a quadratic in `x`, which 2-point Gauss–Legendre
integrates exactly. Summing over the tetrahedra whose `(y, z)` shadow contains
the point gives `Q(y₀, z₀)`.

An earlier version L²-projected onto a bilinear `(y, z)` grid instead. It was
abandoned because the target basis has kinks *inside* the tetrahedra, which no
reasonable quadrature integrates accurately: even with 120 points per cell the
projected width of a box came out 1.6% wrong, and the pointwise values were off
by up to 3% of `max|Q|`.

## Sampling

`ny`, `nz` default to 8 samples per element (`median_edge_length`) in each
direction. That is plenty to render the quadratic variation within an element;
there is no aliasing concern in sampling finer, since every sample is exact.
Outer samples are pulled inside `y_lims`/`z_lims` by a relative `1e-9`, so that
lines on the domain boundary never lie exactly in a mesh face, where adjacent
tetrahedra would both claim the same segment.

`b` is the buoyancy as stored, i.e. the deviation from the background `N² z`;
pass `N²` (the `Model` method does this automatically) to integrate the total
buoyancy instead.
"""
function zonal_integral(fe_data::FEData, state::State, field::Symbol;
                        ny = nothing, nz = nothing,
                        y_lims = nothing, z_lims = nothing, N² = 0.0)
    field in (:u, :v, :w, :p, :b) ||
        throw(ArgumentError("field must be one of :u, :v, :w, :p, :b, got :$field"))

    grid   = fe_data.mesh.grid
    y_lims = something(y_lims, _grid_bbox(grid, 2))
    z_lims = something(z_lims, _grid_bbox(grid, 3))
    if ny === nothing || nz === nothing
        h  = median_edge_length(fe_data.mesh)
        ny = something(ny, max(2, round(Int, 8*(y_lims[2] - y_lims[1])/h) + 1))
        nz = something(nz, max(2, round(Int, 8*(z_lims[2] - z_lims[1])/h) + 1))
    end
    shrink(lims) = (lims[1] + 1e-9*(lims[2] - lims[1]), lims[2] - 1e-9*(lims[2] - lims[1]))
    y = collect(range(shrink(y_lims)...; length = ny))
    z = collect(range(shrink(z_lims)...; length = nz))
    @info @sprintf("Zonal integral of :%s at %d × %d points, y ∈ [%.4g, %.4g], z ∈ [%.4g, %.4g]",
                   field, ny, nz, y_lims..., z_lims...)

    if field === :b
        order  = fe_data.b_order
        dh     = fe_data.dh_b
        x_full = N² == 0 ? state.b : state.b .+ N² .* b_dof_z(fe_data)
        ldofs  = collect(1:ndofs_per_cell(dh))
    else
        dh     = fe_data.dh_up
        x_full = zeros(ndofs(dh))
        x_full[fe_data.u_dof_indices] .= state.u
        x_full[fe_data.p_dof_indices] .= state.p
        if field === :p
            order = fe_data.u_order - 1
            ldofs = collect(dof_range(dh, :p))
        else
            # vector u-DOFs interleave xyz: local DOF 3(j-1)+k is component k of node j
            order = fe_data.u_order
            comp  = field === :u ? 1 : field === :v ? 2 : 3
            ldofs = collect(dof_range(dh, :u))[3 .* (0:getnbasefunctions(
                        Lagrange{RefTetrahedron, order}()) - 1) .+ comp]
        end
    end
    order ≤ 3 || throw(ArgumentError("2-point Gauss is exact only up to degree 3, got order $order"))
    ip = Lagrange{RefTetrahedron, order}()

    Q     = zeros(ny, nz)
    width = zeros(ny, nz)
    @ctime "  line integrals" _zonal_sweep!(Q, width, dh, ip, x_full, ldofs, y, z)
    return (; y, z, Q, width, y_lims, z_lims)
end

function _zonal_sweep!(Q, width, dh, ip, x_full, ldofs, y, z)
    nφ  = getnbasefunctions(ip)
    qe  = zeros(nφ)
    φ   = zeros(nφ)
    gξ  = (-1/√3, 1/√3)        # 2-point Gauss–Legendre on [-1, 1], unit weights
    for cc in CellIterator(dh)
        X    = getcoordinates(cc)
        dofs = celldofs(cc)
        @inbounds for i in 1:nφ
            qe[i] = x_full[dofs[ldofs[i]]]
        end

        # affine map x = X₁ + J ξ, with J's columns the edges from vertex 1
        J    = Tensor{2, 3}(((X[2] - X[1])..., (X[3] - X[1])..., (X[4] - X[1])...))
        Jinv = inv(J)
        c1   = Jinv ⋅ Vec{3}((1.0, 0.0, 0.0))   # dξ/dx along the line

        # sample points inside the (y, z) bounding box of the tetrahedron
        ylo, yhi = extrema(v -> v[2], X)
        zlo, zhi = extrema(v -> v[3], X)
        j1, j2 = searchsortedfirst(y, ylo), searchsortedlast(y, yhi)
        k1, k2 = searchsortedfirst(z, zlo), searchsortedlast(z, zhi)
        for k in k1:k2, j in j1:j2
            c0 = Jinv ⋅ (Vec{3}((0.0, y[j], z[k])) - X[1])
            x_in, x_out = _x_line_interval(c0, c1)
            x_out > x_in || continue
            half = (x_out - x_in)/2
            mid  = (x_out + x_in)/2
            s = 0.0
            for g in gξ
                ξ = c0 + c1*(mid + half*g)
                for i in 1:nφ
                    φ[i] = Ferrite.reference_shape_value(ip, ξ, i)
                end
                s += dot(φ, qe)
            end
            @inbounds Q[j, k]     += s*half
            @inbounds width[j, k] += 2half
        end
    end
    return nothing
end

zonal_integral(m::Model, field::Symbol; kwargs...) =
    zonal_integral(m.fe_data, m.state, field; N² = m.params.N², kwargs...)

"""
    res_ψ = overturning_streamfunction(res_v, α)

Overturning streamfunction

    ψ(y, z) = -1/α ∫_{-H}^{z} ∫ v dx dz′

from `res_v = zonal_integral(..., :v)`. Returns `res_v` with `Q` replaced by `ψ`,
so it plots with [`plot_zonal_integral`](@ref) like any other zonal integral.

The z integral is a cumulative trapezoid rule over the samples, starting from the
bottom of the sample grid; below topography `∫ v dx = 0`, so that is the same as
starting from `-H(y)`. Its error is `O(Δz²)`, small at the default 8 samples per
element.

`ψ` does not return to zero at the surface: `ψ(y, 0) = -T(y)/α`, with `T` the
net transport across the section, and Taylor--Hood (P2/P1) does not make `T`
vanish. The discrete continuity equation zeroes `∫ q ∇·u` only for P1 `q`, so it
conserves volume across *smeared* sections (`q` the P1 interpolant of the step
`H(y - y₀)`), not across the exact plane `y = y₀`. The difference is the
pointwise divergence in the one band of cells straddling the section, and it
alternates in sign from element to element; that is also what draws the
element-scale vertical streaks in `ψ`. Measured on channel_basin_009_cc_su
(h = 4e-2): max|T| ≈ 12% of max ∫|v| dx dz with a direct LU solve, while the
smeared-section transport is 1e-14 (scratch/check_transport.jl). It vanishes
under refinement, at about h^3.6–3.9 for a fixed smooth b
(scratch/transport_convergence.jl).
"""
function overturning_streamfunction(res_v, α)
    V, z = res_v.Q, res_v.z
    ψ = zeros(size(V))
    for k in 2:length(z)
        @views ψ[:, k] .= ψ[:, k-1] .+ (V[:, k] .+ V[:, k-1]) .* (z[k] - z[k-1]) ./ 2
    end
    ψ ./= -α
    return merge(res_v, (; Q = ψ))
end
