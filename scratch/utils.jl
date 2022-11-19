### utility function for some of the stokes/laplace/pg tests

import nuPGCM: tplot
using WriteVTK

function get_sides(g::FEGrid)
    # bottom
    ebot = g.e[abs.(g.p[g.e, 2]) .>= 1e-4]
    eleft = g.e[abs.(g.p[g.e, 1] .+ 1) .<= 1e-4]
    eright = g.e[abs.(g.p[g.e, 1] .- 1) .<= 1e-4]
    ebot = [eleft[1]; ebot; eright[1]]

    # top
    etop = g.e[abs.(g.p[g.e, 2]) .< 1e-4]
    deleteat!(etop, findall(x->x==eleft[1], etop))
    deleteat!(etop, findall(x->x==eright[1], etop))
    return ebot, etop
end

function tplot(u::FEField)
    if u.order == 0
        return tplot(u.g1.p, u.g1.t, u.values)
    else
        return tplot(u.g.p, u.g.t, u.values)
    end
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

function write_vtk(fname, u::FEField, uname)
    # define points and cells for vtk
    points = u.g.p'
    cells = Vector{MeshCell}([])
    for i in axes(u.g.t, 1)
        if u.g.dim == 2
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, u.g.t[i, :]))
        elseif u.g.dim == 3
            push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, u.g.t[i, :]))
        end
    end

    # save as vtu file
    vtk_grid(fname, points, cells) do vtk
        vtk[uname] = u.values
    end
end
function write_vtk(fname, g::FEGrid)
    # define points and cells for vtk
    points = Array(g.p')
    cells = Vector{MeshCell}([])
    for i in axes(g.t, 1)
        if g.dim == 2
            push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, g.t[i, :]))
        elseif g.dim == 3
            push!(cells, MeshCell(VTKCellTypes.VTK_TETRA, g.t[i, :]))
        end
    end

    # save as vtu file
    vtk_grid(fname, points, cells) do vtk
        # boundary = zeros(g.np)
        # boundary[g.e] .= 1
        # vtk["boundary"] = boundary
        # interior = ones(g.np)
        # interior[g.e] .= 0
        # vtk["interior"] = interior
    end
end

# function count_ni(g)
#     ni = 0
#     nb = 0
#     for i in axes(g.t, 1)
#         bdy = false
#         for j=1:4
#             if g.t[i, j] in g.e
#                 bdy = true
#                 nb += 1
#                 break
#             end
#         end
#         if !bdy
#             ni += 1
#         end
#     end
#     println(ni)
#     println(nb)
#     println(g.nt)
# end