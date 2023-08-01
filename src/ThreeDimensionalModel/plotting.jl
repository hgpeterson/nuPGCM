function quick_plot_save(fname, ax)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    ax.set_yticks(-1:0.5:1)
    savefig(fname)
    println(fname)
    plt.close()
end
function quick_plot(u::FEField, cb_label, fname; vmax=nothing)
    fig, ax, im = tplot(u, contour=true, vmax=vmax, cb_label=cb_label)
    quick_plot_save(fname, ax)
end
function quick_plot(u::FVField, cb_label, fname; vmax=nothing)
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

function plot_slice(m::ModelSetup3D, s::ModelState3D, u::AbstractField; cb_label="", fname="slice.png")
    u = FEField(u)
    nx = 2^5
    nσ = 2^5
    x = range(-0.999, 0.999, length=nx)
    y = 0.0
    σ = range(-1., 0., length=nσ)
    Hs = [m.H([x[i], y]) for i ∈ eachindex(x)] 
    xx = repeat(x', nσ, 1)
    zz = repeat(σ, 1, nx).*repeat(Hs', nσ, 1)
    us = [u([x[j], y, σ[i]]) for i ∈ eachindex(σ), j ∈ eachindex(x)]
    bs = [s.b([x[j], y, σ[i]]) for i ∈ eachindex(σ), j ∈ eachindex(x)]
    vmax = maximum(abs.(us))

    fig, ax = plt.subplots(1, figsize=(3.2, 2))
    img = ax.pcolormesh(xx, zz, us, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true, shading="gouraud")
    levels = range(-vmax, vmax, length=8)
    ax.contour(xx, zz, us, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = colorbar(img, ax=ax, label=cb_label)
    levels = range(-1, 0, length=20)
    ax.contour(xx, zz, bs, levels=levels, colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
    ax.fill_between(xx[1, :], zz[1, :], minimum(zz), color="k", alpha=0.3, lw=0.0)
    ax.axis("equal")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig("$out_folder/$fname")
    println("$out_folder/$fname")
    plt.close()
end