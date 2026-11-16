"""
    lims = grid_bbox(grid, dim)

Min/max node coordinate of the mesh along `dim` (1 = x, 2 = y, 3 = z).
"""
function grid_bbox(grid, dim::Int)
    lo, hi = Inf, -Inf
    for node in grid.nodes
        c  = node.x[dim]
        lo = min(lo, c)
        hi = max(hi, c)
    end
    return (lo, hi)
end

"""
    x_in, x_out = x_line_interval(c0, c1)

Interval of `x` over which a line in `x` lies inside a tetrahedron, given the
reference coordinates `ξ(x) = c0 + c1 x` along it. Empty if `x_out ≤ x_in`.
"""
@inline function x_line_interval(c0::Vec{3}, c1::Vec{3})
    x_in, x_out = -Inf, Inf
    # each barycentric coordinate a + b x must be nonnegative
    for (a, b) in ((c0[1], c1[1]), (c0[2], c1[2]), (c0[3], c1[3]),
                   (1 - c0[1] - c0[2] - c0[3], -c1[1] - c1[2] - c1[3]))
        if b > 0
            x_in  = max(x_in, -a/b)
        elseif b < 0
            x_out = min(x_out, -a/b)
        elseif a < 0
            return (Inf, -Inf)
        end
    end
    return (x_in, x_out)
end

"""
    res = zonal_integral(model, field; ny, nz, y_lims, z_lims)
    res = zonal_integral(fe_data, state, field; ny, nz, y_lims, z_lims, N²)

Zonal integral of a field on a structured `(y, z)` grid,

    Q(y, z) = ∫ q(x, y, z) dx,

where `field` is `:u`, `:v`, `:w`, `:p` or `:b`. The integral along each line in
`x` is exact for the finite-element field. `:b` is the total buoyancy `b + N² z`.

Returns a NamedTuple with
- `y`, `z`: sample coordinates, `ny` and `nz` of them (8 per element by default)
  spanning `y_lims` and `z_lims` (the extent of the mesh by default)
- `Q`: the `ny × nz` zonal integral
- `width`: `∫ dx` along the same lines, zero outside the domain
- `y_lims`, `z_lims`
"""
function zonal_integral(fe_data::FEData, state::State, field::Symbol;
                        ny = nothing, nz = nothing,
                        y_lims = nothing, z_lims = nothing, N² = 0.0)
    field in (:u, :v, :w, :p, :b) ||
        throw(ArgumentError("field must be one of :u, :v, :w, :p, :b, got :$field"))

    grid   = fe_data.mesh.grid
    y_lims = something(y_lims, grid_bbox(grid, 2))
    z_lims = something(z_lims, grid_bbox(grid, 3))
    if ny === nothing || nz === nothing
        h  = median_edge_length(fe_data.mesh)
        ny = something(ny, max(2, round(Int, 8*(y_lims[2] - y_lims[1])/h) + 1))
        nz = something(nz, max(2, round(Int, 8*(z_lims[2] - z_lims[1])/h) + 1))
    end
    # keep the outermost lines off the boundary faces of the mesh
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
            # local velocity DOF 3(j-1)+k is component k of node j
            order = fe_data.u_order
            comp  = field === :u ? 1 : field === :v ? 2 : 3
            ldofs = collect(dof_range(dh, :u))[3 .* (0:getnbasefunctions(
                        Lagrange{RefTetrahedron, order}()) - 1) .+ comp]
        end
    end
    ip = Lagrange{RefTetrahedron, order}()

    Q     = zeros(ny, nz)
    width = zeros(ny, nz)
    @ctime "  line integrals" zonal_sweep!(Q, width, dh, ip, x_full, ldofs, y, z)
    return (; y, z, Q, width, y_lims, z_lims)
end

function zonal_sweep!(Q, width, dh, ip, x_full, ldofs, y, z)
    nφ  = getnbasefunctions(ip)
    qe  = zeros(nφ)
    φ   = zeros(nφ)
    gξ  = (-1/√3, 1/√3)        # 2-point Gauss–Legendre on [-1, 1], exact for cubics
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
            x_in, x_out = x_line_interval(c0, c1)
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

from `res_v = zonal_integral(..., :v)`, by the trapezoidal rule in `z`. Returns
`res_v` with `Q` replaced by `ψ`.

`ψ(y, 0)` is small but not zero: P2/P1 elements conserve volume across a section
only in a weak sense.
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
