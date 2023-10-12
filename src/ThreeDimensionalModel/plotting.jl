"""
    fig, ax, im = tplot(p, t, u)

Plot filled contour color plot of solution `u` on mesh defined by nodes positions `p` and connectivities `t`.
"""
function tplot(p, t, u; cmap="RdBu_r", vmax=0., contour=false, cb_label="", cb_orientation="vertical")
    fig, ax = subplots(1)

    # set vmax
    if vmax == 0.
        vmax = maximum(abs.(u))
        extend = "neither"
    else
        # set extend
        if maximum(u) > vmax && minimum(u) < -vmax
            extend = "both"
        elseif maximum(u) > vmax && minimum(u) > -vmax
            extend = "max"
        elseif maximum(u) < vmax && minimum(u) < -vmax
            extend = "min"
        else
            extend = "neither"
        end
    end

    if size(u, 1) == size(t, 1)
        # `u` represents values on triangle faces
        shading = "flat"
    elseif size(u, 1) == size(p, 1)
        # `u` represents values on triangle vertices
        shading = "gouraud"
    end

    im = ax.tripcolor(p[:, 1], p[:, 2], t[:, 1:3] .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading=shading, rasterized=true)
    if contour
        levels = vmax*[-3/4, -1/2, -1/4, 1/4, 1/2, 3/4]
        ax.tricontour(p[:, 1], p[:, 2], t[:, 1:3] .- 1, u, colors="k", linewidths=0.5, linestyles="-", levels=levels)
    end
    cb = colorbar(im, ax=ax, label=cb_label, extend=extend, orientation=cb_orientation)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end
function tplot(u::AbstractField; kwargs...)
    return tplot(u.g.p, u.g.t, u.values; kwargs...)
end

"""
    fig, ax, im = tplot(p, t)

Plot triangular mesh with nodes `p` and triangles `t`.
"""
function tplot(p, t; lw=0.2, edgecolors="k")
    fig, ax = subplots(1)
    im = ax.tripcolor(p[:, 1], p[:, 2], t[:, 1:3] .- 1, 0*t[:, 1], cmap="Greys", edgecolors=edgecolors, linewidth=lw, rasterized=true)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end
function tplot(g::Grid; kwargs...)
    return tplot(g.p, g.t; kwargs...)
end

function quick_plot(u::FEField, cb_label, fname; vmax=0., contour=true)
    fig, ax, im = tplot(u, contour=contour, vmax=vmax, cb_label=cb_label)
    quick_plot_save(fname, ax)
end
function quick_plot(u::FVField, cb_label, fname; vmax=0., contour=false)
    fig, ax, im = tplot(u, contour=contour, vmax=vmax, cb_label=cb_label)
    quick_plot_save(fname, ax)
end
function quick_plot(u::DGField, args...; kwargs...)
    quick_plot(FEField(u), args..., kwargs...)
end
function quick_plot(f::Function, g::Grid, args...; kwargs...)
    quick_plot(FEField(f, g), args...; kwargs...)
end
function quick_plot_save(fname, ax)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Meridional coordinate $y$")
    ax.axis("equal")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:1)
    savefig(fname)
    println(fname)
    plt.close()
end

function write_vtk(g, fname, data)
    # define points and cells for vtk
    points = g.p'

    # cells
    if g.el <: Triangle 
        if g.el.n == 3
            cell_type = VTKCellTypes.VTK_TRIANGLE
        elseif g.el.n == 6
            cell_type = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
        end
    elseif g.el <: Wedge
        cell_type = VTKCellTypes.VTK_WEDGE
    end
    cells = [MeshCell(cell_type, g.t[i, :]) for i ∈ axes(g.t, 1)]

    # save as vtu file
    vtk_grid(fname, points, cells) do vtk
        for d ∈ data
            if typeof(d.second) <: AbstractField
                vtk[d.first] = d.second.values
            else
                vtk[d.first] = d.second
            end
        end
    end
    println(fname)
end

function plot_ω_χ(m, ωx, ωy, χx, χy; fname="$out_folder/omega_chi.vtu")
    # unpack 
    g = m.g1
    g_sfc2 = m.g_sfc2
    H = m.H
    nσ = m.nσ

    # DG p, t
    np = g.nt*g.nn
    p = zeros(Float64, (np, 3))
    t = zeros(Int64, (g.nt, 6))

    # global solutions
    ωx_plot = zeros(np)
    ωy_plot = zeros(np)
    χx_plot = zeros(np)
    χy_plot = zeros(np)

    # all the nodes within each column will have a unique tag
    i_p = 0
    for k_sfc=1:g_sfc2.nt
        for j=1:nσ-1
            k_w = get_k_w(k_sfc, nσ, j)
            p[i_p+1:i_p+6, 1:2] = g.p[g.t[k_w, :], 1:2]
            p[i_p+1:i_p+3, 3] = g.p[g.t[k_w, 1:3], 3].*H[g_sfc2.t[k_sfc, 1:3]]
            p[i_p+4:i_p+6, 3] = g.p[g.t[k_w, 4:6], 3].*H[g_sfc2.t[k_sfc, 1:3]]
            t[k_w, :] = i_p+1:i_p+6
            ωx_plot[i_p+1:i_p+6] = ωx[k_w, :]
            ωy_plot[i_p+1:i_p+6] = ωy[k_w, :]
            χx_plot[i_p+1:i_p+6] = χx[k_w, :]
            χy_plot[i_p+1:i_p+6] = χy[k_w, :]
            i_p += 6
        end
    end

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_WEDGE, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid(fname, p', cells) do vtk
        vtk["omega^x"] = ωx_plot
        vtk["omega^y"] = ωy_plot
        vtk["chi^x"] = χx_plot
        vtk["chi^y"] = χy_plot
    end
    println(fname)
end

function plot_xslice(m::ModelSetup3D, b::AbstractField, u::AbstractField, y, cb_label, fname)
    # params
    nx = 2^8
    nσ = m.nσ
    σ = m.σ

    # get x slice
    bdy = m.g_sfc1.p[m.g_sfc1.e["bdy"], :]
    neary = sort(bdy[sortperm(abs.(bdy[:, 2] .- y)), 1][1:4])
    x = range(neary[2], neary[3], length=nx)
    
    # get indices of surface tris
    k_sfcs = [get_k([x[i], y], m.g_sfc1, m.g_sfc1.el) for i=1:nx]

    # get points in reference tri
    ξ_sfcs = [transform_to_ref_el(m.g_sfc1.el, [x[i], y], m.g_sfc1.p[m.g_sfc1.t[k_sfcs[i], :], :]) for i=1:nx]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[j], nσ, i) for i=1:nσ-1, j=1:nx]
    k_ws = vcat(k_ws, k_ws[end, :]')

    # get points in reference wedge
    ξ_ws = [transform_to_ref_el(m.g1.el, [x[j], y, σ[i]], m.g1.p[m.g1.t[k_ws[i, j], :], :]) for i=1:nσ, j=1:nx]

    # nσ × nx coords
    Hs = [m.H(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    xx = repeat(x', nσ, 1)
    zz = repeat(σ, 1, nx).*repeat(Hs', nσ, 1)

    # evaluate
    u_fe = FEField(u)
    us = [u_fe(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]
    bs = [b(ξ_ws[i, j], k_ws[i, j])   for i=1:nσ, j=1:nx]

    # plot
    title = latexstring(@sprintf("Slice at \$y = %1.1f\$", y))
    plot_vertical_slice(xx, zz, us, bs, cb_label, fname, title, slice_dir="x")
end

function plot_yslice(m::ModelSetup3D, b::AbstractField, u::AbstractField, x, cb_label, fname)
    # params
    ny = 2^8
    nσ = m.nσ
    σ = m.σ

    # get y slice
    bdy = m.g_sfc1.p[m.g_sfc1.e["bdy"], :]
    nearx = sort(bdy[sortperm(abs.(bdy[:, 1] .- x)), 2][1:4])
    y = range(nearx[2], nearx[3], length=ny)

    # get indices of surface tris
    k_sfcs = [get_k([x, y[i]], m.g_sfc1, m.g_sfc1.el) for i=1:ny]

    # get points in reference tri
    ξ_sfcs = [transform_to_ref_el(m.g_sfc1.el, [x, y[i]], m.g_sfc1.p[m.g_sfc1.t[k_sfcs[i], :], :]) for i=1:ny]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[j], nσ, i) for i=1:nσ-1, j=1:ny]
    k_ws = vcat(k_ws, k_ws[end, :]')

    # get points in reference wedge
    ξ_ws = [transform_to_ref_el(m.g1.el, [x, y[j], σ[i]], m.g1.p[m.g1.t[k_ws[i, j], :], :]) for i=1:nσ, j=1:ny]

    # nσ × ny coords
    Hs = [m.H(ξ_sfcs[i], k_sfcs[i]) for i=1:ny] 
    yy = repeat(y', nσ, 1)
    zz = repeat(σ, 1, ny).*repeat(Hs', nσ, 1)

    # evaluate
    u_fe = FEField(u)
    us = [u_fe(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:ny]
    bs = [b(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:ny]

    # plot
    title = latexstring(@sprintf("Slice at \$x = %1.1f\$", x))
    plot_vertical_slice(yy, zz, us, bs, cb_label, fname, title, slice_dir="y")
end

function plot_zslice(m::ModelSetup3D, u::AbstractField, z, cb_label, fname)
    g = m.g_sfc1
    H = m.H

    u_fe = FEField(u)
    u_slice = zeros(g.np)
    for i=1:g.np
        if H[i] < abs(z)
            u_slice[i] = NaN
        else
            u_slice[i] = u_fe([g.p[i, 1], g.p[i, 2], z/H[i]])
        end
    end

    mask = [any(isnan.(u_slice[g.t[k, :]])) for k=1:g.nt]
    vmax = maximum(i-> isnan(u_slice[i]) ? -Inf : u_slice[i], 1:g.np)

    # plot
    title = latexstring(@sprintf("Slice at \$z = %1.1f\$", z))
    fig, ax = plt.subplots(1)
    img = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u_slice, mask=mask, cmap="rdbu_r", vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=cb_label)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), usemathtext=true)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Meridional coordinate $y$")
    ax.axis("equal")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:1)
    ax.set_title(title)
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_u(m::ModelSetup3D, s::ModelState3D, y)
    # params
    nx = 2^8
    nσ = m.nσ
    σ = m.σ

    # get x slice
    bdy = m.g_sfc1.p[m.g_sfc1.e["bdy"], :]
    neary = sort(bdy[sortperm(abs.(bdy[:, 2] .- y)), 1][1:4])
    x = range(neary[2], neary[3], length=nx)
    
    # get indices of surface tris
    k_sfcs = [get_k([x[i], y], m.g_sfc1, m.g_sfc1.el) for i=1:nx]

    # get points in reference tri
    ξ_sfcs = [transform_to_ref_el(m.g_sfc1.el, [x[i], y], m.g_sfc1.p[m.g_sfc1.t[k_sfcs[i], :], :]) for i=1:nx]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[j], nσ, i) for i=1:nσ-1, j=1:nx]
    k_ws = vcat(k_ws, k_ws[end, :]')

    # get points in reference wedge
    ξ_ws = [transform_to_ref_el(m.g1.el, [x[j], y, σ[i]], m.g1.p[m.g1.t[k_ws[i, j], :], :]) for i=1:nσ, j=1:nx]

    # nσ × nx coords
    Hs = [m.H(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    Hxs = [m.Hx(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    Hys = [m.Hy(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    xx = repeat(x', nσ, 1)
    zz = repeat(σ, 1, nx).*repeat(Hs', nσ, 1)

    # evaluate
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    χx = [χx_fe(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]
    χy = [χy_fe(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]
    ux = zeros(nσ, nx)
    uy = zeros(nσ, nx)
    for i=1:nx
        ux[:, i] = -differentiate(χy[:, i], σ*Hs[i])
        uy[:, i] = +differentiate(χx[:, i], σ*Hs[i])
    end
    Huσ = [∂x(χy_fe, ξ_ws[i, j], k_ws[i, j]) - ∂y(χx_fe, ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]
    uz = [Huσ[i, j] + σ[i]*Hxs[j]*ux[i, j] + σ[i]*Hys[j]*uy[i, j] for i=1:nσ, j=1:nx]
    bs = [s.b(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]

    # plot
    title = latexstring(@sprintf("Slice at \$y = %1.1f\$", y))
    plot_vertical_slice(xx, zz, ux, bs, L"Zonal flow $u^x$",      "$out_folder/ux.png", title, contour=false, slice_dir="x")
    plot_vertical_slice(xx, zz, uy, bs, L"Meridional flow $u^y$", "$out_folder/uy.png", title, contour=false, slice_dir="x")
    plot_vertical_slice(xx, zz, uz, bs, L"Vertical flow $u^z$",   "$out_folder/uz.png", title, contour=false, slice_dir="x")
end

function plot_vertical_slice(xx, zz, u, b, cb_label, fname, title; contour=true, slice_dir)
    fig, ax = plt.subplots(1)
    vmax = maximum(abs.(u))
    img = ax.pcolormesh(xx, zz, u, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true, shading="gouraud")
    if contour
        levels = range(-vmax, vmax, length=8)
        ax.contour(xx, zz, u, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    end
    cb = colorbar(img, ax=ax, label=cb_label, fraction=0.0235)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    levels = range(-1, 0, length=20)
    ax.contour(xx, zz, b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    if slice_dir == "x"
        ax.set_xlabel(L"Zonal coordinate $x$")
    elseif slice_dir == "y"
        ax.set_xlabel(L"Meridional coordinate $y$")
    end
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.set_title(title)
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_profiles(m::ModelSetup3D, b, ωx, ωy, χx, χy, x, y, fname)
    k_sfc = get_k([x, y], m.g_sfc1, m.g_sfc1.el)
    ξ_sfc = transform_to_ref_el(m.g_sfc1.el, [x, y], m.g_sfc1.p[m.g_sfc1.t[k_sfc, :], :])

    σ = m.σ
    nσ = m.nσ
    H = m.H(ξ_sfc, k_sfc)
    z = σ*H
    k_ws = get_k_ws(k_sfc, nσ)
    k_ws = [k_ws; k_ws[end]]
    ξ_ws = [transform_to_ref_el(m.g1.el, [x, y, σ[i]], m.g1.p[m.g1.t[k_ws[i], :], :]) for i=1:nσ]

    ωx_fe = FEField(ωx)
    ωy_fe = FEField(ωy)
    χx_fe = FEField(χx)
    χy_fe = FEField(χy)
    ωxs = [ωx_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    ωys = [ωy_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    χxs = [χx_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    χys = [χy_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    bs = [b(ξ_ws[i], k_ws[i]) for i=1:nσ]
    bzs = differentiate(bs, z)

    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)
    ax[1, 1].plot(ωxs, z)
    ax[1, 2].plot(ωys, z)
    ax[1, 3].plot(bs, z)
    ax[2, 1].plot(χxs, z)
    ax[2, 2].plot(χys, z)
    ax[2, 3].plot(bzs, z)
    ax[1, 1].set_xlabel(L"\omega^x")
    ax[1, 2].set_xlabel(L"\omega^y")
    ax[1, 3].set_xlabel(L"b")
    ax[2, 1].set_xlabel(L"\chi^x")
    ax[2, 2].set_xlabel(L"\chi^y")
    ax[2, 3].set_xlabel(L"\partial_z b")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 2].set_title(latexstring(@sprintf("\$x = %1.1f \\quad y = %1.1f\$", x, y)))
    ax[1, 1].set_ylim(-H, 0)
    ax[2, 1].set_ylim(-H, 0)
    for a ∈ ax
        a.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    end
    savefig(fname)
    println(fname)
    plt.close()
end
function plot_profiles(m::ModelSetup3D, s::ModelState3D, args...; kwargs...)
    plot_profiles(m, s.b, s.ωx, s.ωy, s.χx, s.χy, args...; kwargs...)
end