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
    zonal_integral(m.fe_data, on_architecture(CPU(), m.state), field; N² = m.params.N², kwargs...)

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

"""
    drift = drift_diagnostics(fe_data, params, files; isopycnals, y_band = (-0.5, 0.5),
                              bottom_fraction = 0.1, channel = (-Inf, -0.5),
                              overturning_stride = 50)

Drift of the states saved by `save_state` in `files`, in time order. With the total
buoyancy `b + N² z`, the result has, for each state `k`,

- `t[k]`: model time
- `b_mean[k]`: volume mean of the buoyancy
- `b_bottom[k]`: mean buoyancy over the bottom `bottom_fraction` of the domain height
- `isopycnal_depth[k, i]`: mean depth of `b = isopycnals[i]` over `y_band`, the volume
  of lighter water there divided by the horizontal area
- `kinetic_energy[k]`, `potential_energy[k]`: volume means of `|u|²/2` and `-b z`
- `buoyancy_flux[k]`: net flux of buoyancy into the domain, `d/dt ∫ b dΩ` between
  states `k - 1` and `k`, `NaN` for the first state
- `channel_transport[k]`: zonal-mean zonal transport `∫ u dΩ / Lₓ` over the `y` range
  `channel`, with `Lₓ` the zonal extent of the domain
- `ψ_max[k]`, `ψ_min[k]`: extremes of the overturning streamfunction, for every
  `overturning_stride`-th state and `NaN` otherwise
- `change[k, :]`: `‖Xₖ - Xₖ₋₁‖ / (‖Xₖ‖ (tₖ - tₖ₋₁))` for `X = u, p, b` and the whole
  state `(u, p, b)`, `NaN` for the first state
"""
function drift_diagnostics(fe_data::FEData, params::Parameters, files; isopycnals,
                           y_band = (-0.5, 0.5), bottom_fraction = 0.1,
                           channel = (-Inf, -0.5), overturning_stride = 50)
    cache = fe_data.cache
    nq, ncells = length(cache.w), length(cache.detJ)
    n_b, n_nodes = size(cache.phi_b, 2), size(cache.phi_u, 2)

    # values at the quadrature points, ordered (q, c), are Q * DOFs
    point(q, c) = q + nq*(c - 1)
    Qb = sparse([point(q, c) for i in 1:n_b, q in 1:nq, c in 1:ncells][:],
                [cache.dofs_b[i, c] for i in 1:n_b, q in 1:nq, c in 1:ncells][:],
                [cache.phi_b[q, i] for i in 1:n_b, q in 1:nq, c in 1:ncells][:],
                nq*ncells, fe_data.nb)
    Qu = [sparse([point(q, c) for j in 1:n_nodes, q in 1:nq, c in 1:ncells][:],
                 [cache.dofs_u[3(j - 1) + k, c] for j in 1:n_nodes, q in 1:nq, c in 1:ncells][:],
                 [cache.phi_u[q, j] for j in 1:n_nodes, q in 1:nq, c in 1:ncells][:],
                 nq*ncells, fe_data.nu) for k in 1:3]
    x  = [cache.x0[c] + cache.J[c] ⋅ cache.x_ref[q] for q in 1:nq, c in 1:ncells][:]
    dΩ = [cache.w[q] * cache.detJ[c] for q in 1:nq, c in 1:ncells][:]
    z  = getindex.(x, 3)

    x_lims = grid_bbox(fe_data.mesh.grid, 1)
    z_lims = grid_bbox(fe_data.mesh.grid, 3)
    bottom_weights  = dΩ .* (z .< z_lims[1] + bottom_fraction*(z_lims[2] - z_lims[1]))
    band_weights    = dΩ .* (y_band[1] .< getindex.(x, 2) .< y_band[2])
    band_area       = sum(band_weights ./ params.H.(x))   # ∫ dz/H = 1 over each column
    channel_weights = dΩ .* (channel[1] .< getindex.(x, 2) .< channel[2]) ./ (x_lims[2] - x_lims[1])

    n = length(files)
    t = zeros(n); b_mean = zeros(n); b_bottom = zeros(n)
    kinetic_energy = zeros(n); potential_energy = zeros(n); channel_transport = zeros(n)
    isopycnal_depth = zeros(n, length(isopycnals))
    buoyancy_flux = fill(NaN, n); ψ_max = fill(NaN, n); ψ_min = fill(NaN, n)
    change = fill(NaN, n, 4)
    previous = nothing
    for (k, file) in enumerate(files)
        u, p, b, t[k] = load(file, "u", "p", "b", "t")
        b_q = Qb*b .+ params.N² .* z
        b_mean[k]   = dot(dΩ, b_q) / sum(dΩ)
        b_bottom[k] = dot(bottom_weights, b_q) / sum(bottom_weights)
        for (i, bᵢ) in enumerate(isopycnals)
            isopycnal_depth[k, i] = dot(band_weights, b_q .> bᵢ) / band_area
        end
        u1, u2, u3 = (Q*u for Q in Qu)
        kinetic_energy[k]    = dot(dΩ, u1.^2 .+ u2.^2 .+ u3.^2) / (2sum(dΩ))
        potential_energy[k]  = -dot(dΩ, b_q .* z) / sum(dΩ)
        channel_transport[k] = dot(channel_weights, u1)
        if mod(k - 1, overturning_stride) == 0
            ψ = overturning_streamfunction(zonal_integral(fe_data, State(u, p, b), :v), params.α).Q
            ψ_max[k], ψ_min[k] = maximum(ψ), minimum(ψ)
        end
        if previous !== nothing
            buoyancy_flux[k] = sum(dΩ) * (b_mean[k] - previous.b_mean) / (t[k] - previous.t)
            Δ² = [sum(abs2, X - Y) for (X, Y) in zip((u, p, b), previous.X)]
            X² = [sum(abs2, X) for X in (u, p, b)]
            Δt = t[k] - previous.t
            change[k, 1:3] .= sqrt.(Δ² ./ X²) ./ Δt
            change[k, 4]    = sqrt(sum(Δ²) / sum(X²)) / Δt
        end
        previous = (; X = (u, p, b), t = t[k], b_mean = b_mean[k])
    end
    return (; t, b_mean, b_bottom, isopycnals, isopycnal_depth, kinetic_energy,
              potential_energy, buoyancy_flux, channel_transport, ψ_max, ψ_min, change)
end

"""
    plot_drift(drifts, labels; ofile = "\$out_dir/images/drift")

Plot the `drift_diagnostics` results `drifts` of the runs named `labels` against time,
one panel per metric, and save the figure to `ofile.png` and `ofile.pdf`.
"""
function plot_drift(drifts, labels; ofile = "$out_dir/images/drift")
    pc = 1/6   # pica [in]
    fig, ax = subplots(3, 3, figsize=(33pc, 27pc), sharex=true)
    for (a, letter) in zip(permutedims(ax), 'a':'i')
        a.annotate("($letter)", xy=(-0.04, 1.05), xycoords="axes fraction")
    end
    # dashes keep runs that coincide visible
    for (j, (d, label)) in enumerate(zip(drifts, labels))
        c, ls = "C$(j - 1)", ("-", "--", ":")[mod1(j, 3)]
        ax[1, 1].plot(d.t, d.b_mean; c, ls, label)
        ax[1, 2].plot(d.t, d.b_bottom; c, ls)
        for i in eachindex(d.isopycnals)
            ax[1, 3].plot(d.t, -d.isopycnal_depth[:, i]; c, ls)
        end
        ax[2, 1].plot(d.t, d.kinetic_energy; c, ls)
        ax[2, 2].plot(d.t, d.potential_energy; c, ls)
        ax[2, 3].plot(d.t, d.channel_transport; c, ls)
        ax[3, 1].semilogy(d.t, abs.(d.buoyancy_flux); c, ls)
        sampled = .!isnan.(d.ψ_max)
        ax[3, 2].plot(d.t[sampled], d.ψ_max[sampled]; c, ls)
        ax[3, 2].plot(d.t[sampled], d.ψ_min[sampled]; c, ls)
        ax[3, 3].semilogy(d.t, d.change[:, 4]; c, ls)
    end
    # name the isopycnals at the end of the longest run
    d = drifts[argmax([last(d.t) for d in drifts])]
    for (i, bᵢ) in enumerate(d.isopycnals)
        ax[1, 3].text(last(d.t), -d.isopycnal_depth[end, i], latexstring(@sprintf(" \$b = %g\$", bᵢ));
                      va="center", ha="left")
    end
    sampled = findlast(!isnan, d.ψ_max)
    ax[3, 2].text(d.t[sampled], d.ψ_max[sampled], L"$\psi_\mathrm{max}$"; va="bottom", ha="right")
    ax[3, 2].text(d.t[sampled], d.ψ_min[sampled], L"$\psi_\mathrm{min}$"; va="bottom", ha="right")
    ax[1, 1].set_ylabel(L"Mean buoyancy $\langle b \rangle$")
    ax[1, 2].set_ylabel(L"Bottom buoyancy $\langle b \rangle_\mathrm{bot}$")
    ax[1, 3].set_ylabel(L"Isopycnal height $z$")
    ax[2, 1].set_ylabel(L"Kinetic energy $\langle |\mathbf{u}|^2 \rangle / 2$")
    ax[2, 2].set_ylabel(L"Potential energy $-\langle b z \rangle$")
    ax[2, 3].set_ylabel(L"Channel transport $T$")
    ax[3, 1].set_ylabel(L"Net buoyancy flux $|d_t \int b \, d\Omega|$")
    ax[3, 2].set_ylabel(L"Overturning $\psi$")
    ax[3, 3].set_ylabel(L"State change $\|\Delta X\| / (\|X\| \Delta t)$")
    for a in ax[3, :]
        a.set_xlabel(L"Time $t$")
    end
    ax[1, 1].legend(loc="lower right")
    subplots_adjust(wspace=0.6, hspace=0.3)

    for ext in ("png", "pdf")
        fig.savefig("$ofile.$ext")
        @info "Drift plot saved to '$ofile.$ext'"
    end
    close(fig)
    return nothing
end

"""
    plot_state_change(fe_data, file₁, file₂; y, z, n = 200, ofile = "\$out_dir/images/state_change")

Map `|u₂ - u₁|` between the states saved by `save_state` in `file₁` and `file₂`, on the
horizontal plane at height `z` and on the vertical plane at `y`, sampled at `n` points
across the domain, and save the figure to `ofile.png` and `ofile.pdf`.
"""
function plot_state_change(fe_data::FEData, file₁, file₂; y, z, n = 200,
                           ofile = "$out_dir/images/state_change")
    grid = fe_data.mesh.grid
    x_lims, y_lims, z_lims = (grid_bbox(grid, dim) for dim in 1:3)
    u₁, t₁ = load(file₁, "u", "t")
    u₂, t₂ = load(file₂, "u", "t")
    Δu = to_up_vec(fe_data, u₂ - u₁)
    speed(points) = [isnothing(v) ? NaN : norm(v) for v in
                     evaluate_at_points(PointEvalHandler(grid, points; warn=false), fe_data.dh_up, Δu, :u)]
    xs = range(x_lims...; length=n)
    ys = range(y_lims...; length=n)
    zs = range(z_lims...; length=n ÷ 4)
    horizontal = reshape(speed([Vec{3}((x′, y′, z)) for x′ in xs for y′ in ys]), n, n)
    vertical   = reshape(speed([Vec{3}((x′, y, z′)) for x′ in xs for z′ in zs]), length(zs), n)
    vmax = maximum(filter(!isnan, [vec(horizontal); vec(vertical)]))

    pc = 1/6   # pica [in]
    fig, ax = subplots(1, 2, figsize=(33pc, 12pc))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    img = ax[1].pcolormesh(xs, ys, horizontal; shading="nearest", cmap="viridis", vmin=0, vmax, rasterized=true)
    ax[2].pcolormesh(xs, zs, vertical; shading="nearest", cmap="viridis", vmin=0, vmax, rasterized=true)
    ax[1].set_title(latexstring(@sprintf("\$z = %g\$", z)))
    ax[2].set_title(latexstring(@sprintf("\$y = %g\$", y)))
    ax[1].set_xlabel(L"Zonal coordinate $x$")
    ax[1].set_ylabel(L"Meridional coordinate $y$")
    ax[2].set_xlabel(L"Zonal coordinate $x$")
    ax[2].set_ylabel(L"Vertical coordinate $z$")
    for a in ax
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    fig.colorbar(img, ax=ax[2], label=latexstring(@sprintf("\$|\\mathbf{u}(t = %.4g) - \\mathbf{u}(t = %.4g)|\$", t₂, t₁)))
    subplots_adjust(wspace=0.3)

    for ext in ("png", "pdf")
        fig.savefig("$ofile.$ext")
        @info "State change plot saved to '$ofile.$ext'"
    end
    close(fig)
    return nothing
end
