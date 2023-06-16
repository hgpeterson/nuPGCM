function quick_plot(u::Union{FEField,FVField}, cb_label, fname; vmax=nothing)
    contour = !(typeof(u) <: FVField)
    fig, ax, im = tplot(u, contour=contour, vmax=vmax, cb_label=cb_label)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig(fname)
    println(fname)
    plt.close()
end
function quick_plot(u::DGField, args...; kwargs...)
    u_plot = zeros(u.g.np)
    count = zeros(u.g.np)
    for k=1:u.g.nt
        for i=1:u.g.nn
            u_plot[u.g.t[k, i]] += u[k, i]
            count[u.g.t[k, i]] += 1
        end
    end
    u_plot ./= count
    u_plot = FEField(u_plot, u.g)
    quick_plot(u_plot, args...; kwargs...)
end
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
    if g.order == 0 
        cell_type = VTKCellTypes.VTK_VERTEX
    elseif g.order == 1
        cell_type = VTKCellTypes.VTK_TETRA
    elseif g.order == 2
        cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
    end
    cells = [MeshCell(cell_type, g.t[i, :]) for i ∈ axes(g.t, 1)]

    # save as vtu file
    vtk_grid(fname, points, cells) do vtk
        for d ∈ data
            if typeof(d.second) <: FEField
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
        ωx_plot[i_p+1:i_p+g.np] = ωx[k]
        ωy_plot[i_p+1:i_p+g.np] = ωy[k]
        χx_plot[i_p+1:i_p+g.np] = χx[k]
        χy_plot[i_p+1:i_p+g.np] = χy[k]

        # increment
        i_p += g.np
        i_t += g.nt
    end

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid("output/omega_chi.vtu", p', cells) do vtk
        vtk["omegaˣ"] = ωx_plot
        vtk["omegaʸ"] = ωy_plot
        vtk["chiˣ"] = χx_plot
        vtk["chiʸ"] = χy_plot
    end
    println("output/omega_chi.vtu")
end