####
#### Field evaluation helpers (replace Gridap CellField evaluation)
####

"""
    vals = eval_at_points(model, dh, dof_vec, field, points)

Evaluate a scalar FE field at a list of `Vec{3}` points using Ferrite's
`PointEvalHandler`. Returns `NaN` for points outside the domain.
"""
function eval_at_points(model::Model, dh, dof_vec::AbstractVector,
                         field::Symbol, points::Vector{Vec{3, Float64}})
    grid = model.fe_data.mesh.grid
    ph   = PointEvalHandler(grid, points; warn=false)
    vals = evaluate_at_points(ph, dh, dof_vec, field)
    return [isnothing(v) ? NaN : v for v in vals]
end

"""
    val = eval_at_point(model, dh, dof_vec, field, x::Vec{3})

Evaluate a scalar FE field at a single point. Returns `NaN` if outside the domain.
"""
function eval_at_point(model::Model, dh, dof_vec::AbstractVector,
                        field::Symbol, x::Vec{3, Float64})
    v = eval_at_points(model, dh, dof_vec, field, [x])
    return v[1]
end

"""
    H = find_H(model, x, y; tol=1e-8)

Find the depth of the water column at (x, y) by bisection using the buoyancy field.
"""
function find_H(model::Model, x::Real, y::Real; tol=1e-8)
    z_in  = 0.0
    z_out = -1.0
    while abs(z_in - z_out) > tol
        z = (z_in + z_out) / 2
        if isnan(eval_at_point(model, model.fe_data.dh_b, model.state.b,
                                :b, Vec{3}((Float64(x), Float64(y), z))))
            z_out = z
        else
            z_in = z
        end
    end
    return -z_in
end

####
#### Sparsity pattern visualisation
####

"""
    plot_sparsity_pattern(model)

Plot the sparsity pattern of the inversion matrix.
"""
function plot_sparsity_pattern(model::Model)
    A = on_architecture(CPU(), model.inversion.solver.A)
    I, J, _ = findnz(A)
    fig, ax = subplots()
    ax.spy(A, markersize=0.5)
    ax.set_title("Inversion matrix sparsity pattern ($(size(A,1)) × $(size(A,2)), nnz=$(nnz(A)))")
    return fig, ax
end

####
#### Zonally integrated fields
####

# smallest of 1, 2, 2.5, 5 × 10ᵏ that is ≥ x > 0
function _nice_ceil(x)
    e = 10.0^floor(log10(x))
    for m in (1, 2, 2.5, 5, 10)
        m*e ≥ x*(1 - 1e-12) && return m*e
    end
end

"""
    fig, ax = plot_zonal_integral(res; kwargs...)

Colour plot of a [`zonal_integral`](@ref) result (or an
[`overturning_streamfunction`](@ref)) in the `(y, z)` plane.

`res` is the NamedTuple returned by `zonal_integral`. Keyword arguments:
- `label`       — colorbar label (default `""`)
- `title`       — axes title (default `""`)
- `ofile`       — path to save to; `nothing` (default) leaves the figure open
- `cmap`        — colormap (default `"RdBu_r"`, a diverging map)
- `vmax`        — colour saturation; defaults to `maximum(abs, Q)` rounded up to
                  1, 2, 2.5 or 5 × 10ᵏ so that the colorbar ends on a tick.
                  Symmetric limits centre the diverging map on zero, which is
                  what makes the sign of the circulation readable
- `contours`    — number of intervals for overlaid solid black contour lines of `Q`,
                  `0` for none (default). The zero line and ±`vmax` are left
                  out: where the field is weak the zero line traces noise and
                  swamps the plot
- `isopycnals`  — a `zonal_integral(..., :b)` result; its zonal *mean*
                  `Q ./ width` is drawn as faint black contours (default `nothing`)
- `n_isopycnals`— number of isopycnals, evenly spaced strictly inside the range
                  of the zonal-mean `b` (default 20)
- `mask_tol`    — points whose wet width is below `mask_tol * maximum(width)`
                  are blanked out; this is what hides the region under
                  topography (default `1e-3`)
- `figsize`     — figure size in inches; the default is wide and short because
                  these domains are much wider than they are deep (8:1 for the
                  channel-basin box), so a true aspect ratio squashes the layers

Note that `z` is *not* plotted to scale — see `figsize` above.
"""
function plot_zonal_integral(res;
                             label = "", title = "", ofile = nothing,
                             cmap = "RdBu_r", vmax = nothing, contours::Int = 0,
                             isopycnals = nothing, n_isopycnals::Int = 20,
                             mask_tol = 1e-3, figsize = (3.4, 1.5))
    dry = res.width .≤ mask_tol * maximum(res.width)
    Q = copy(res.Q)
    Q[dry] .= NaN

    if vmax === nothing
        vmax = maximum(x -> isnan(x) ? -Inf : abs(x), Q)
        vmax = vmax > 0 ? _nice_ceil(vmax) : 1.0
    end

    fig, ax = subplots(1, 1; figsize)
    # samples are nodal values, so Gouraud shading interpolates them exactly like
    # the plotted field should look; it cannot mask NaN triangles cleanly, so fall
    # back to one flat cell per sample when there is a dry region.
    # rasterized: ~10⁴ quads would otherwise bloat vector output
    img = ax.pcolormesh(res.y, res.z, permutedims(Q);
                        cmap, vmin = -vmax, vmax = vmax, rasterized = true,
                        shading = any(dry) ? "nearest" : "gouraud")
    if contours > 0
        lines = filter(l -> 1e-12vmax < abs(l) < (1 - 1e-12)vmax,
                       collect(range(-vmax, vmax; length = contours + 1)))
        ax.contour(res.y, res.z, permutedims(Q), lines;
                   colors = "k", linestyles = "-", linewidths = 0.25)
    end
    if isopycnals !== nothing
        b̄ = isopycnals.Q ./ isopycnals.width
        b̄[isopycnals.width .≤ mask_tol * maximum(isopycnals.width)] .= NaN
        blo, bhi = extrema(filter(!isnan, b̄))
        if bhi > blo
            lvls = collect(range(blo, bhi; length = n_isopycnals + 2))[2:end-1]
            ax.contour(isopycnals.y, isopycnals.z, permutedims(b̄), lvls;
                       colors = "k", linestyles = "-", linewidths = 0.5, alpha = 0.3)
        end
    end
    fig.colorbar(img, ax = ax, label = label,
                 ticks = matplotlib.ticker.MaxNLocator(nbins = 4, symmetric = true))
    ax.set_xlabel(L"Meridional coordinate $y$")
    ax.set_ylabel(L"$z$")
    ax.set_xlim(res.y_lims...)
    ax.set_ylim(res.z_lims...)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    isempty(title) || ax.set_title(title)

    if ofile !== nothing
        fig.savefig(ofile; bbox_inches = "tight")
        @info "Zonal integral plot saved to '$ofile'"
        close(fig)
    end
    return fig, ax
end

####
#### Simulation-level plots — stub implementations
####
# These are placeholders; detailed slice/profile plots require postprocessing
# the VTK output in ParaView or a dedicated postprocessing script.

function plot_slice(args...; kwargs...)
    @warn "plot_slice not yet implemented for Ferrite backend; use save_vtk + ParaView"
    return nothing
end

function plot_profiles(args...; kwargs...)
    @warn "plot_profiles not yet implemented for Ferrite backend"
    return nothing
end

function sim_plots(model::Model, t::Real; kwargs...)
    @warn "sim_plots not yet implemented for Ferrite backend"
    return nothing
end

function nan_eval(args...; kwargs...)
    @warn "nan_eval has been replaced by eval_at_point / eval_at_points"
    return NaN
end
