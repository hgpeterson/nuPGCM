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

function plot_ω_χ(m, ωx, ωy, χx, χy)
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
            k_w = (k_sfc - 1)*(nσ - 1) + j
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

    # # map solutions from [k_sfc, i_sfc, j] to [k_wedge, i_wedge]
    # ωx_plot = zeros(g.nt, g.nn)
    # ωy_plot = zeros(g.nt, g.nn)
    # χx_plot = zeros(g.nt, g.nn)
    # χy_plot = zeros(g.nt, g.nn)
    # nσ = size(ωx, 3)
    # for k_sfc ∈ axes(ωx, 1)
    #     for i ∈ axes(ωx, 2)
    #         for j=1:nσ-1
    #             k_w = (k_sfc - 1)*(nσ - 1) + j
    #             ωx_plot[k_w, i] = ωx[k_sfc, i, j]
    #             ωy_plot[k_w, i] = ωy[k_sfc, i, j]
    #             χx_plot[k_w, i] = χx[k_sfc, i, j]
    #             χy_plot[k_w, i] = χy[k_sfc, i, j]
    #             ωx_plot[k_w, i+3] = ωx[k_sfc, i, j+1]
    #             ωy_plot[k_w, i+3] = ωy[k_sfc, i, j+1]
    #             χx_plot[k_w, i+3] = χx[k_sfc, i, j+1]
    #             χy_plot[k_w, i+3] = χy[k_sfc, i, j+1]
    #         end
    #     end
    # end

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_WEDGE, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid("output/omega_chi.vtu", p', cells) do vtk
        vtk["omega^x"] = ωx_plot
        vtk["omega^y"] = ωy_plot
        vtk["chi^x"] = χx_plot
        vtk["chi^y"] = χy_plot
    end
    println("output/omega_chi.vtu")
end