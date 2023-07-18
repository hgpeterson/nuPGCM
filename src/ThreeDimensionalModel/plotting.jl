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
    quick_plot(FVField(u), args..., kwargs...)
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

function plot_ω_χ(ωx, ωy, χx, χy, g_cols)
    # global p, t, e
    np = sum(g.np for g ∈ g_cols)
    nt = sum(g.nt for g ∈ g_cols)
    p = zeros(Float64, (np, 3))
    t = zeros(Int64, (nt, 4))

    # global solutions
    ωx_plot = zeros(np)
    ωy_plot = zeros(np)
    χx_plot = zeros(np)
    χy_plot = zeros(np)

    # current indices
    i_p = 0
    i_t = 0

    # all the nodes within each column will have a unique tag
    for k ∈ eachindex(g_cols)
        # column
        g = g_cols[k]

        # add nodes, triangles, and edge nodes
        p[i_p+1:i_p+g.np, :] = g.p
        t[i_t+1:i_t+g.nt, :] = i_p .+ g.t

        # unpack solutions
        ωx_plot[i_p+1:i_p+g.np] = vcat(ωx[k, 1], ωx[k, 2], ωx[k, 3])
        ωy_plot[i_p+1:i_p+g.np] = vcat(ωy[k, 1], ωy[k, 2], ωy[k, 3])
        χx_plot[i_p+1:i_p+g.np] = vcat(χx[k, 1], χx[k, 2], χx[k, 3])
        χy_plot[i_p+1:i_p+g.np] = vcat(χy[k, 1], χy[k, 2], χy[k, 3])

        # increment
        i_p += g.np
        i_t += g.nt
    end

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid("output/omega_chi.vtu", p', cells) do vtk
        vtk["omega^x"] = ωx_plot
        vtk["omega^y"] = ωy_plot
        vtk["chi^x"] = χx_plot
        vtk["chi^y"] = χy_plot
    end
    println("output/omega_chi.vtu")
end