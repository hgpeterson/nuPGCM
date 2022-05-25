"""
    fig, ax, im = tplot(p, t, u=nothing)

If `u` === nothing: Plot triangular mesh with nodes `p` and triangles `t`.
If `u` === solution vector: Plot filled contour color plot of solution `u`.
"""
function tplot(p, t, u=nothing; ax=nothing, cmap="RdBu_r", vext=nothing)
    if ax === nothing
        fig, ax = subplots(1)
    end

    if u === nothing
        im = ax.tripcolor(p[:, 1], p[:, 2], t .- 1, 0*t[:, 1], cmap="Set3", edgecolors="k", linewidth=0.5)
    else
        if vext === nothing
            vmax = maximum(abs.(u))
        else
            vmax = vext
        end

        if size(u, 1) == size(t, 1)
            # `u` represents values on triangle faces
            shading = "flat"
        elseif size(u, 1) == size(p, 1)
            # `u` represents values on triangle vertices
            shading = "gouraud"
        end

        im = ax.tripcolor(p[:, 1], p[:, 2], t .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading=shading)
    end

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end

function plot_horizontal(p, t, u; vext=nothing, clabel="", contours=true)
    if vext === nothing
        vext = maximum(abs.(u))
        extend = "neither"
    else
        extend = "both"
    end
    p = p/1e3 # km
    fig, ax, im = tplot(p, t, u; vext=vext)
    cb = colorbar(im, ax=ax, label=clabel, extend=extend)
    if contours
        n = 6
        levels = vext*[collect(-(n-1)/n:1/n:-1/n)' collect(1/n:1/n:(n-1)/n)']
        ax.tricontour(p[:, 1], p[:, 2], t .- 1, u, linewidths=0.25, colors="k", linestyles="-", levels=levels)
    end
    ax.set_xlabel(L"Horizontal coordinate $\xi$ (km)")
    ax.set_ylabel(L"Horizontal coordinate $\eta$ (km)")
    ax.set_yticks(-5000:2500:5000)
    ax.axis("equal")
end

function plot_ξ_slice(m::ModelSetup3DPG, s::ModelState3DPG, v::AbstractArray{<:Real,2},
                      ξ_range::AbstractArray{<:Real,1}, η₀::Real; vext=nothing, clabel="", contours=true)
    # compute v, b, and H on ξ range
    nξ = size(ξ_range, 1)
    v_slice = zeros(nξ, m.nσ)
    b_slice = zeros(nξ, m.nσ)
    H_slice = zeros(nξ)
    @showprogress "Computing slices for plotting..." for i=1:nξ
        for j=1:m.nσ
            v_slice[i, j] = fem_evaluate(m, v[:, j],   ξ_range[i], η₀)
            b_slice[i, j] = fem_evaluate(m, s.b[:, j], ξ_range[i], η₀)
        end
        H_slice[i] = fem_evaluate(m ,m.H, ξ_range[i], η₀)
    end

    # init figure
    fig, ax = subplots()
    ax.set_xlabel(L"Horizontal coordinate $\xi$ (km)")
    ax.set_ylabel(L"Vertical coordinate $z$ (km)")

    # call plotting function
    return plot_slice(m, ax, ξ_range, v_slice, b_slice, H_slice; vext=vext, clabel=clabel, contours=contours)
end

function plot_η_slice(m::ModelSetup3DPG, s::ModelState3DPG, v::AbstractArray{<:Real,2},
                      η_range::AbstractArray{<:Real,1}, ξ₀::Real; vext=nothing, clabel="", contours=true)
    # compute v, b, and H on η range
    nη = size(η_range, 1)
    v_slice = zeros(nη, m.nσ)
    b_slice = zeros(nη, m.nσ)
    H_slice = zeros(nη)
    @showprogress "Computing slices for plotting..." for i=1:nη
        for j=1:m.nσ
            v_slice[i, j] = fem_evaluate(m, v[:, j],   ξ₀, η_range[i])
            b_slice[i, j] = fem_evaluate(m, s.b[:, j], ξ₀, η_range[i])
        end
        H_slice[i] = fem_evaluate(m, m.H, ξ₀, η_range[i])
    end

    # init figure
    fig, ax = subplots()
    ax.set_xlabel(L"Horizontal coordinate $\eta$ (km)")
    ax.set_ylabel(L"Vertical coordinate $z$ (km)")

    # call plotting function
    return plot_slice(m, ax, η_range, v_slice, b_slice, H_slice; vext=vext, clabel=clabel, contours=contours)
end

function plot_slice(m::ModelSetup3DPG, ax, x::AbstractArray{<:Real,1}, v_slice::AbstractArray{<:Real,2}, b_slice::AbstractArray{<:Real,2},
                    H_slice::AbstractArray{<:Real,1}; vext=nothing, clabel="", contours=true)
    # z coordinates along slice 
    nx = size(x, 1)
    z = repeat(m.σ', nx, 1).*repeat(H_slice, 1, m.nσ)

    # extremes
    if vext === nothing
        vext = maximum(abs.(v_slice))
        extend = "neither"
    else
        extend = "both"
    end

    # plot data
    xx = repeat(x, 1, m.nσ)
    img = ax.pcolormesh(xx/1e3, z/1e3, v_slice, cmap="RdBu_r", vmin=-vext, vmax=vext, rasterized=true, shading="auto")
    if contours
        levels = range(-vext, vext, length=8)
        ax.contour(xx/1e3, z/1e3, v_slice, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    end
    cb = colorbar(img, ax=ax, label=clabel, extend=extend)

    # isopycnal contours
    n_levels = 20
    i = argmax(m.H)
    lower_level = -trapz(m.N²[i, :], m.H[i]*m.σ)
    upper_level = lower_level/100
    levels = lower_level:(upper_level - lower_level)/(n_levels - 1):upper_level
    ax.contour(xx/1e3, z/1e3, b_slice, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)

    # topo
    ax.fill_between(x/1e3, z[:, 1]/1e3, minimum(z)/1e3, color="k", alpha=0.3, lw=0.0)

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)

    return ax
end