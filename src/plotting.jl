####
#### Field evaluation
####

"""
    vals = eval_at_points(model, dh, dof_vec, field, points)

Evaluate the field `field` of the DOF vector `dof_vec` on `dh` at a list of
`Vec{3}` points. Returns `NaN` for points outside the domain.
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

Evaluate a field at a single point. Returns `NaN` if outside the domain.
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
        if isnan(eval_at_point(model, model.fe_data.dh_b, on_architecture(CPU(), model.state.b),
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
    fig, ax = subplots()
    ax.spy(A, markersize=0.5)
    ax.set_title("Inversion matrix sparsity pattern ($(size(A,1)) × $(size(A,2)), nnz=$(nnz(A)))")
    return fig, ax
end

####
#### Zonally integrated fields
####

# smallest of 1, 2, 2.5, 5 × 10ᵏ that is ≥ x > 0
function nice_ceil(x)
    e = 10.0^floor(log10(x))
    for m in (1, 2, 2.5, 5, 10)
        m*e ≥ x*(1 - 1e-12) && return m*e
    end
end

"""
    fig, ax = plot_zonal_integral(res; kwargs...)

Colour plot in the `(y, z)` plane of a result `res` of [`zonal_integral`](@ref) or
[`overturning_streamfunction`](@ref).

- `label`, `title`: colorbar label and axes title
- `ofile`: file to save the figure to; if `nothing`, the figure is left open
- `cmap`: colormap
- `vmax`: colour limits are `±vmax`; by default `maximum(abs, Q)` rounded up to
  1, 2, 2.5 or 5 × 10ᵏ
- `contours`: number of contour intervals of `Q` between `±vmax`; the zero
  contour is not drawn
- `isopycnals`: a `zonal_integral(..., :b)` result, whose zonal mean is contoured
  at `n_isopycnals` levels
- `mask_tol`: points where the domain is narrower than `mask_tol * maximum(width)`
  are blanked
- `figsize`: figure size in inches; `z` is not drawn to scale
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
        vmax = vmax > 0 ? nice_ceil(vmax) : 1.0
    end

    fig, ax = subplots(1, 1; figsize)
    # Gouraud shading interpolates between samples but cannot mask NaNs
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
