using LinearAlgebra
using Gmsh: gmsh
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

"""
    node_coords, connectivities = get_nodes_connectivities_bottom(mesh)

Get the coordinates of the nodes and the connectivities of the triangles for 
the bottom surface of the `mesh`.
"""
function get_nodes_connectivities_bottom(mesh)
    gmsh.initialize()
    gmsh.open(mesh)

    bot_tag = 1 # tag of bottom boundary physical group

    # get coordinates for _every_ node in the mesh
    node_tags, node_coords, parametric_coords = gmsh.model.mesh.get_nodes()
    node_coords = Matrix(reshape(node_coords, 3, :)') # N × 3 matrix

    # get triangle connectivities for every triangle on the bottom boundary
    bot_entity_tags = gmsh.model.get_entities_for_physical_group(2, bot_tag)
    connectivities = []
    for tag in bot_entity_tags
        element_types, element_tags, node_tags = gmsh.model.mesh.get_elements(2, tag)
        push!(connectivities, Matrix(reshape(Int.(node_tags[1]), 3, :)'))
    end
    connectivities = vcat(connectivities...)

    gmsh.finalize()

    # indices of nodes on the bottom surface
    bottom_nodes = sort(unique(connectivities[:]))

    # filter node_coords to only include bottom nodes
    node_coords = node_coords[bottom_nodes, :]

    # adjust indices in connectivities match the new node_coords
    imap = Dict(bottom_nodes[i] => i for i in eachindex(bottom_nodes))
    connectivities = [imap[connectivities[k, i]] for k in axes(connectivities, 1), i in axes(connectivities, 2)]

    return node_coords, connectivities
end

# """
#     node_to_triangles = get_node_to_triangles(node_coords, connectivities)

# Create a mapping from each node to the triangles it is part of.
# """
# function get_node_to_triangles(node_coords, connectivities)
#     node_to_triangles = [CartesianIndex[] for _ in 1:size(node_coords, 1)]
#     for k in axes(connectivities, 1), i in axes(connectivities, 2)
#         push!(node_to_triangles[connectivities[k, i]], CartesianIndex(k, i))
#     end 
#     return node_to_triangles
# end

"""
    get_centroids(node_coords, connectivities)

Compute the centroids of the triangles in the (x, y) plane.
"""
function get_centroids(node_coords, connectivities)
    centroids = zeros(size(connectivities, 1), 2)
    for k in axes(connectivities, 1)
        vertices = node_coords[connectivities[k, :], 1:2] # (x, y) only
        centroids[k, :] = (vertices[1, :] + vertices[2, :] + vertices[3, :])/3
    end
    return centroids
end

"""
    k = get_triangle(x, node_coords, connectivities, centroids)

Find the index `k` of the triangle that contains the point `x`.
"""
function get_triangle(x, node_coords, connectivities, centroids)
    # convert Gridap VectorValue to Vector 
    x = [x[1], x[2]]

    # squared distance from point `x` to each triangle centroid
    dists = [(centroids[k, 1] - x[1])^2 + (centroids[k, 2] - x[2])^2 for k in axes(centroids, 1)]

    # check if the point is inside each triangle in order of distance to centroid
    ordered_indices = sortperm(dists)
    for k in ordered_indices
        if is_point_in_triangle(x, node_coords[connectivities[k, :], 1:2])
            return k # return the triangle index
        end
    end

    error("Point $(x) is not inside any triangle.")
end

"""
    bool = is_point_in_triangle(x, p)

Determine if point `x` is in triangle with nodes `p`.
(See https://stackoverflow.com/a/2049593).
"""
function is_point_in_triangle(x, p)
    d₁ = pt_sign(x, p[1, :], p[2, :])
    d₂ = pt_sign(x, p[2, :], p[3, :])
    d₃ = pt_sign(x, p[3, :], p[1, :])

    has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
    has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

    return !(has_neg && has_pos)
end
function pt_sign(p₁, p₂, p₃)
    return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
end

"""
    eval_triangle(x, vertices, values)

Evaluate the linear interpolation of `values` at point `x` inside the triangle
defined by `vertices`.
"""
function eval_triangle(x, vertices, values)
    # linear function: z = c1 + c2*x + c3*y
    # -> [1 x1 y1; 1 x2 y2; 1 x3 y3] * [c1; c2; c3] = [z1; z2; z3]
    V = [1 vertices[1, 1] vertices[1, 2];
         1 vertices[2, 1] vertices[2, 2];
         1 vertices[3, 1] vertices[3, 2]]
    c = V \ values # solve for coefficients
    # evaluate at x
    return c[1] + c[2]*x[1] + c[3]*x[2]
end

################################################################################

node_coords, connectivities = get_nodes_connectivities_bottom("../meshes/channel_basin_cart.msh")
# node_to_triangles = get_node_to_triangles(node_coords, connectivities)
centroids = get_centroids(node_coords, connectivities)
function H(x)
    k = get_triangle(x, node_coords, connectivities, centroids)
    vertices = node_coords[connectivities[k, :], 1:2]
    values = -node_coords[connectivities[k, :], 3] # depth is -z
    return eval_triangle(x, vertices, values)
end

# using PyPlot
# pygui(false)
# plt.style.use("../plots.mplstyle")
# plt.close("all")

# fig, ax = plt.subplots(1)
# x = range(0, 1, length=100)
# y = -0.7:0.001:-0.5
# αs = range(1, 0, length=length(y))
# for i in eachindex(y)
#     z = [H([x[j], y[i]]) for j in eachindex(x)]
#     ax.plot(x, z, c="C0", alpha=αs[i], label=latexstring("\$y = ", y[i], "\$"))
# end
# ax.set_xlabel(L"x")
# ax.set_ylabel(L"H")
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 0.75)
# # ax.legend()
# savefig("channel_basin_depth.png")
# @info "Saved 'channel_basin_depth.png'"
# plt.close()

# H([0.020202020202020204, -0.55])

# fig, ax = plt.subplots(1, figsize=(4, 8))
# for k in axes(connectivities, 1)
#     vertices = node_coords[connectivities[k, :], 1:2]
#     ax.fill(vertices[:, 1], vertices[:, 2], alpha=0.3)
#     ax.plot(centroids[k, 1], centroids[k, 2], "k.", markersize=1)
# end
# ax.set_xlabel(L"x")
# ax.set_ylabel(L"y")
# ax.set_xlim(0, 1)
# ax.set_ylim(-1, 1)
# savefig("centroids.png")
# @info "Saved 'centroids.png'"
# plt.close()