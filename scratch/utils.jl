### utility function for some of the stokes/laplace/pg tests

using WriteVTK

function get_sides(g::FEGrid; tol=1e-4)
    # bottom
    bot = g.e[abs.(g.p[g.e, end]) .≥ tol]

    # top
    sfc = g.e[abs.(g.p[g.e, end]) .< tol]

    if g.dim == 2
        # for 2D, add corners to bot and remove them from sfc
        left = g.e[abs.(g.p[g.e, 1] .+ 1) .≤ tol]
        right = g.e[abs.(g.p[g.e, 1] .- 1) .≤ tol]
        bot = [left[1]; bot; right[1]]
    elseif g.dim == 3
        # for 3D, add coastline to bot
        coast = g.e[abs.(1 .- sqrt.(g.p[g.e, 1].^2 + g.p[g.e, 2].^2)) .≤ 1e2*tol]
        bot = [bot; coast]
    end

    # println(findall(i->i∈bot && i∈sfc, 1:g.np) == sort(coast))

    return bot, sfc
end

function quickplot(u::FEField, clabel, ofile)
    fig, ax, im = tplot(u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end
function quickplot(b::FEField, u::FEField, clabel, ofile)
    fig, ax, im = tplot(u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.tricontour(b.g.p[:, 1], b.g.p[:, 2], b.g.t[:, 1:3] .- 1, b.values, linewidths=0.5, colors="k", linestyles="-", alpha=0.3)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end
function quickplot(x, H, b::FEField, u::FEField, clabel, ofile)
    fig, ax, im = tplot(u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.tricontour(b.g.p[:, 1], b.g.p[:, 2], b.g.t[:, 1:3] .- 1, b.values, linewidths=0.5, colors="k", linestyles="-", alpha=0.3)
    ax.fill_between(x, -maximum(H), -H, color="k", alpha=0.3, lw=0.0)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
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

function add_dirichlet(A, b, row::Integer, col::Integer, u₀::Real)
    # delete row
    A[row, :] .= 0
    # replace Aᵢⱼ = 1
    A[row, col] = 1
    # replace bᵢ = 1
    b[row] = u₀
    # row reduce
    for k in eachindex(b)
        if k == row
            continue
        end
        b[k] -= A[k, col]*u₀
        A[k, col] = 0
    end
    return A, b
end
function add_dirichlet(A, b, row::Integer, u₀::Real)
    return add_dirichlet(A, b, row, row, u₀)
end

function add_dirichlet(A, b, rows::AbstractVector, cols::AbstractVector, u₀::AbstractVector)
    for i in eachindex(rows)
        A, b = add_dirichlet(A, b, rows[i], cols[i], u₀[i])
    end
    return A, b
end
function add_dirichlet(A, b, rows::AbstractVector, cols::AbstractVector, u₀::Real)
    return add_dirichlet(A, b, rows, cols, u₀*ones(size(rows)))
end

function add_dirichlet(A, b, rows::AbstractVector, u₀::AbstractVector)
    return add_dirichlet(A, b, rows, rows, u₀)
end
function add_dirichlet(A, b, rows::AbstractVector, u₀::Real)
    return add_dirichlet(A, b, rows, u₀*ones(size(rows)))
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
        for d in data
            vtk[d.first] = d.second.values
        end
    end
end