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
function tplot(u::FEField; kwargs...)
    return tplot(u.g.p, u.g.t, u.values; kwargs...)
end
function tplot(u::FVField; kwargs...)
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

function quick_plot_save(fname, ax)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    ax.set_yticks(-1:0.5:1)
    savefig(fname)
    println(fname)
    plt.close()
end
function quick_plot(u::FEField, cb_label, fname; vmax=0.)
    fig, ax, im = tplot(u, contour=true, vmax=vmax, cb_label=cb_label)
    quick_plot_save(fname, ax)
end
function quick_plot(u::FVField, cb_label, fname; vmax=0.)
    fig, ax, im = tplot(u, contour=false, vmax=vmax, cb_label=cb_label)
    quick_plot_save(fname, ax)
end
function quick_plot(u::DGField, args...; kwargs...)
    quick_plot(FEField(u), args..., kwargs...)
end
# function quick_plot(u::DGField, cb_label, fname)
#     fig, ax = plt.subplots(1, 2, gridspec_kw=Dict("width_ratios"=>[20, 1]))
#     vmax = maximum(abs(u))
#     g = u.g
#     for k=1:g.nt
#         ax[1].tripcolor(g.p[g.t[k, :], 1], g.p[g.t[k, :], 2], [0 1 2], u[k, 1:3], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=true)
#     end
#     norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
#     cmap = mpl.cm.RdBu_r
#     cb = mpl.colorbar.ColorbarBase(ax[2], norm=norm, cmap=cmap, label=cb_label)
#     cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
#     ax[1].spines["left"].set_visible(false)
#     ax[1].spines["bottom"].set_visible(false)
#     ax[1].set_xlabel(L"x")
#     ax[1].set_ylabel(L"y")
#     ax[1].axis("equal")
#     ax[1].set_yticks(-1:0.5:1)
#     savefig(fname)
#     println(fname)
#     plt.close()
# end
function quick_plot(f::Function, g::Grid, args...; kwargs...)
    quick_plot(FEField(f, g), args...; kwargs...)
end

function plot_profile(u::FEField, x, z, xlabel, ylabel, ofile)
    u_profile = zeros(size(z))
    for i in eachindex(z)
        u_profile[i] = evaluate(u, [x, z[i]])
    end
    fig, ax = subplots(1, figsize=(2, 3.2))
    ax.plot(u_profile, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    savefig(ofile)
    println(ofile)
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

function plot_xslices(m::ModelSetup3D, s::ModelState3D, y; fname="$out_folder/xslices.png")
    # params
    nx = 2^5
    nσ = m.nσ
    σ = m.σ

    # get x slice
    bdy = m.g_sfc1.p[m.g_sfc1.e["bdy"], :]
    neary = sort(bdy[sortperm(abs.(bdy[:, 2] .- y)), 1][1:4])
    x = range(neary[2], neary[3], length=nx)
    
    # get indices of surface tris
    k_sfcs = [get_k([x[i], y], m.g_sfc1, m.g_sfc1.el) for i=1:nx]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[i], nσ, j) for i=1:nx, j=1:nσ-1]
    k_ws = hcat(k_ws, k_ws[:, end])

    # nσ × nx coords
    Hs = [m.H([x[i], y], k_sfcs[i]) for i=1:nx] 
    xx = repeat(x', nσ, 1)
    zz = repeat(σ, 1, nx).*repeat(Hs', nσ, 1)

    # evaluate
    ωx_fe = FEField(s.ωx)
    ωy_fe = FEField(s.ωy)
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    ωx = [ωx_fe([x[j], y, σ[i]], k_ws[j, i]) for i=1:nσ, j=1:nx]
    ωy = [ωy_fe([x[j], y, σ[i]], k_ws[j, i]) for i=1:nσ, j=1:nx]
    χx = [χx_fe([x[j], y, σ[i]], k_ws[j, i]) for i=1:nσ, j=1:nx]
    χy = [χy_fe([x[j], y, σ[i]], k_ws[j, i]) for i=1:nσ, j=1:nx]
    bs = [s.b([x[j], y, σ[i]], k_ws[j, i])   for i=1:nσ, j=1:nx]

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
    plot_slice(ax[1, 1], xx, zz, ωx, bs, L"Vorticity $\omega^x$")
    plot_slice(ax[1, 2], xx, zz, ωy, bs, L"Vorticity $\omega^y$")
    plot_slice(ax[2, 1], xx, zz, χx, bs, L"Streamfunction $\chi^x$")
    plot_slice(ax[2, 2], xx, zz, χy, bs, L"Streamfunction $\chi^y$")
    for a in ax
        a.set_xticks(-1:0.5:1)
        a.set_yticks(-1:0.5:0)
        a.set_xlabel(L"Zonal coordinate $x$")
        a.set_ylabel(L"Vertical coordinate $z$")
    end
    subplots_adjust(hspace=0.3, wspace=0.5)
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_yslices(m::ModelSetup3D, s::ModelState3D, x; fname="$out_folder/yslices.png")
    # params
    ny = 2^5
    nσ = m.nσ
    σ = m.σ

    # get y slice
    bdy = m.g_sfc1.p[m.g_sfc1.e["bdy"], :]
    nearx = sort(bdy[sortperm(abs.(bdy[:, 1] .- x)), 2][1:4])
    y = range(nearx[2], nearx[3], length=ny)
    
    # get indices of surface tris
    k_sfcs = [get_k([x, y[i]], m.g_sfc1, m.g_sfc1.el) for i=1:ny]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[i], nσ, j) for i=1:ny, j=1:nσ-1]
    k_ws = hcat(k_ws, k_ws[:, end])

    # nσ × nx coords
    Hs = [m.H([x, y[i]], k_sfcs[i]) for i=1:ny] 
    yy = repeat(y', nσ, 1)
    zz = repeat(σ, 1, ny).*repeat(Hs', nσ, 1)

    # evaluate
    ωx_fe = FEField(s.ωx)
    ωy_fe = FEField(s.ωy)
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    ωx = [ωx_fe([x, y[j], σ[i]], k_ws[j, i]) for i=1:nσ, j=1:ny]
    ωy = [ωy_fe([x, y[j], σ[i]], k_ws[j, i]) for i=1:nσ, j=1:ny]
    χx = [χx_fe([x, y[j], σ[i]], k_ws[j, i]) for i=1:nσ, j=1:ny]
    χy = [χy_fe([x, y[j], σ[i]], k_ws[j, i]) for i=1:nσ, j=1:ny]
    bs = [s.b([x, y[j], σ[i]], k_ws[j, i])   for i=1:nσ, j=1:ny]

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 4))
    plot_slice(ax[1, 1], yy, zz, ωx, bs, L"Vorticity $\omega^x$")
    plot_slice(ax[1, 2], yy, zz, ωy, bs, L"Vorticity $\omega^y$")
    plot_slice(ax[2, 1], yy, zz, χx, bs, L"Streamfunction $\chi^x$")
    plot_slice(ax[2, 2], yy, zz, χy, bs, L"Streamfunction $\chi^y$")
    for a in ax
        a.set_xticks(-1:0.5:1)
        a.set_yticks(-1:0.5:0)
        a.set_xlabel(L"Meridional coordinate $y$")
        a.set_ylabel(L"Vertical coordinate $z$")
    end
    subplots_adjust(hspace=0.3, wspace=0.5)
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_slice(ax, xx, zz, u, b, cb_label)
    vmax = maximum(abs.(u))
    img = ax.pcolormesh(xx, zz, u, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true, shading="gouraud")
    levels = range(-vmax, vmax, length=8)
    ax.contour(xx, zz, u, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = colorbar(img, ax=ax, label=cb_label, fraction=0.0235)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    levels = range(-1, 0, length=20)
    ax.contour(xx, zz, b, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
    ax.fill_between(xx[1, :], zz[1, :], minimum(zz), color="k", alpha=0.3, lw=0.0)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
end

function plot_profiles(m::ModelSetup3D, s::ModelState3D, x, y; fname="$out_folder/profiles.png")
    k_sfc = get_k([x, y], m.g_sfc1, m.g_sfc1.el)

    σ = m.σ
    nσ = m.nσ
    H = m.H([x, y])
    z = σ*H
    k_ws = get_k_ws(k_sfc, nσ)
    k_ws = [k_ws; k_ws[end]]

    ωx_fe = FEField(s.ωx)
    ωy_fe = FEField(s.ωy)
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    ωx = [ωx_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    ωy = [ωy_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    χx = [χx_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    χy = [χy_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    b = [s.b([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    bz = [∂z(s.b, [x, y, σ[i]], k_ws[i])/H for i=1:nσ]

    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)
    ax[1, 1].plot(ωx, z)
    ax[1, 2].plot(ωy, z)
    ax[1, 3].plot(b, z)
    ax[2, 1].plot(χx, z)
    ax[2, 2].plot(χy, z)
    ax[2, 3].plot(bz, z)
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