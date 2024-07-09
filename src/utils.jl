"""
    x = chebyshev_nodes(n)

Return `n` Chebyshev nodes in the interval `[-1, 1]`.
"""
function chebyshev_nodes(n)
    return [-cos((i - 1)*π/(n - 1)) for i ∈ 1:n]
end

struct MyGrid{P, T, PT}
    p::P
    t::T
    p_to_t::PT
end

"""
    g = MyGrid(model)
    g = MyGrid(fname)
    g = MyGrid(p, t, p_to_t)

A simple custom struct to hold a mesh. `p` defines the node coordinates, 
`t` defines the connectivities, and `p_to_t` maps nodes to cells.
"""
function MyGrid(model::Gridap.Geometry.UnstructuredDiscreteModel)
    p, t = get_p_t(model)
    p_to_t = get_p_to_t(t, size(p, 1))
    return MyGrid(p, t, p_to_t)
end
function MyGrid(fname::String)
    model = GmshDiscreteModel(fname)
    return MyGrid(model)
end

"""
    p, t = get_p_t(model)
    p, t = get_p_t(fname)

Return the node coordinates `p` and the connectivities `t` of a mesh.
"""
function get_p_t(model::Gridap.Geometry.UnstructuredDiscreteModel)
    # unpack node coords
    nc = model.grid.node_coordinates
    np = length(nc)
    d = length(nc[1])
    p = [nc[i][j] for i ∈ 1:np, j ∈ 1:d]

    # unpack connectivities
    cni = model.grid.cell_node_ids
    nt = length(cni)
    nn = length(cni[1])
    t = [cni[i][j] for i ∈ 1:nt, j ∈ 1:nn]

    return p, t
end
function get_p_t(fname::String)
    # load model
    model = GmshDiscreteModel(fname)
    return get_p_t(model)
end

"""
    p_to_t = get_p_to_t(t, np)

Returns a vector of vectors of vectors `p_to_t` such that p_to_t[i] lists
all the [k, j] pairs in `t` that point to the ith node of the mesh of size `np`.
"""
function get_p_to_t(t, np)
    p_to_t = [[] for i ∈ 1:np]
    for k ∈ axes(t, 1)
        for i ∈ axes(t, 2)
            push!(p_to_t[t[k, i]], [k, i])
        end
    end
    return p_to_t
end

"""
    u(x) = nan_eval(u, x)

Evaluate `u(x)` and return `NaN` if an error occurs.
"""
function nan_eval(u::Gridap.CellField, x)
    try u(x) catch NaN end
end

"""
    u = unpack_fefunction(u, g)

Unpack a `FEFunction` `u` into a vector of values `u` at the nodes of the mesh.
(Assumes `u` is continuous).
"""
function unpack_fefunction(u, g::MyGrid)
    u_cell_values = get_cell_dof_values(u)
    return [u_cell_values[g.p_to_t[i][1][1]][g.p_to_t[i][1][2]] for i ∈ 1:size(g.p, 1)]
end

"""
    quick_plot(u, g; b, label, fname)

Plot a scalar field `u` on a mesh `g`.
"""
function quick_plot(u, g::MyGrid; b=nothing, label="", fname="image.png")
    u = unpack_fefunction(u, g)
    fig, ax = plt.subplots(1)
    umax = maximum(abs.(u))
    img = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u, shading="gouraud", vmin=-umax, vmax=umax, cmap="RdBu_r", rasterized=true)
    if b !== nothing
        b = unpack_fefunction(b, g)
        ax.tricontour(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, b, colors="k", linewidths=0.5, linestyles="-", alpha=0.3, levels=-0.95:0.05:-0.05)
    end
    cb = plt.colorbar(img, ax=ax, label=label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    savefig(fname)
    println(fname)
    plt.close()
end