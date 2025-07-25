struct Mesh{MF, MI, DVI, DMI}
    nodes::MF
    elements::MI
    boundary_nodes::DVI
    boundary_elements::DMI
end

"""
    Mesh(meshfile)

Read the data from the Gmsh .msh file `meshfile` and create a `Mesh`. 

Mesh properties
===============

- `nodes`: An ``N × 3`` matrix where ``N`` is the number of nodes. The ``i``th row 
           contains the ``(x, y, z)`` coordinates of node ``i``. 

- `elements`: An ``M × n`` matrix where ``M`` is the number of elements and ``n`` 
              is the number of nodes per element.

- `boundary_nodes`: A dictionary where each key is a boundary name and the values
                    are vectors indices of the nodes on that boundary.

- `boundary_elements`: A dictionary where each key is a boundary name and the values
                       are matrices of node indices on that boundary where each row
                       corresponds to a boundary element.
"""
function Mesh(meshfile)
    gmsh.initialize()
    gmsh.open(meshfile)

    # get coordinates for every node in the mesh
    node_tags, nodes, _ = gmsh.model.mesh.get_nodes()
    nodes = Matrix(reshape(nodes, 3, :)') # N × 3 matrix

    # get all the elementary entities in the model
    # `entities` is as a vector of (`dim`, `tag`) pairs
    entities = gmsh.model.get_entities()
    n_entities = length(entities)

    # number of nodes per interior element
    # if `dim` of last entity is 2, n = 3 (triangle)
    # if `dim` of last entity is 3, n = 4 (tetrahedron)
    n = entities[n_entities][1] + 1

    # interior elements
    dim = entities[n_entities][1] #FIXME: this assumes the last entity is the interior but there could be multiple disconnected regions
    tag = entities[n_entities][2]
    _, _, elements = gmsh.model.mesh.get_elements(dim, tag)
    @assert length(elements) == 1 "Expected exactly one element type in interior."
    elements = Matrix(reshape(Int.(elements[1]), n, :)') # M × n matrix

    # surface and bottom nodes and elements
    boundary_nodes = Dict{String, Vector{Int}}()
    boundary_elements = Dict{String, Matrix{Int}}()
    for i in [n_entities-1, n_entities-2]
        dim = entities[i][1]
        tag = entities[i][2]

        # surface or bottom?
        physical_tags = gmsh.model.get_physical_groups_for_entity(dim, tag)
        @assert length(physical_tags) == 1 "Mesh must have exactly one physical group per entity."
        name = gmsh.model.get_physical_name(dim, physical_tags[1])

        # nodes
        node_tags, _, _ = gmsh.model.mesh.get_nodes(dim, tag)
        boundary_nodes[name] = Int.(node_tags) # convert to Int

        # elements
        _, _, els = gmsh.model.mesh.get_elements(dim, tag)
        @assert length(els) == 1 "Expected exactly one element type on '$name' boundary."
        boundary_elements[name] = Matrix(reshape(Int.(els[1]), n-1, :)') # n_elements × n-1 matrix
    end

    gmsh.finalize()

    return Mesh(nodes, elements, boundary_nodes, boundary_elements)
end

# struct MeshCache{MF, V}
#     centroids::MF
#     surface_centroids::MF
#     bottom_centroids::MF
#     node_to_elements::V
# end

# """
#     node_to_elements = get_node_to_elements(nodes, elements)

# Create a mapping from each node to the elements it is part of.
# """
# function get_node_to_elements(nodes, elements)
#     node_to_elements = [CartesianIndex[] for _ in 1:size(nodes, 1)]
#     for k in axes(elements, 1), i in axes(elements, 2)
#         push!(node_to_elements[elements[k, i]], CartesianIndex(k, i))
#     end 
#     return node_to_elements
# end

# """
#     get_centroids(nodes, elements)

# Compute the centroids of the triangles in the (x, y) plane.
# """
# function get_centroids(nodes, elements)
#     centroids = zeros(size(elements, 1), 2)
#     for k in axes(elements, 1)
#         vertices = nodes[elements[k, :], 1:2] # (x, y) only
#         centroids[k, :] = (vertices[1, :] + vertices[2, :] + vertices[3, :])/3
#     end
#     return centroids
# end

# """
#     k = get_triangle(x, nodes, elements, centroids)

# Find the index `k` of the triangle that contains the point `x`.
# """
# function get_triangle(x, nodes, elements, centroids)
#     # convert Gridap VectorValue to Vector 
#     x = [x[1], x[2]]

#     # squared distance from point `x` to each triangle centroid
#     dists = [(centroids[k, 1] - x[1])^2 + (centroids[k, 2] - x[2])^2 for k in axes(centroids, 1)]

#     # check if the point is inside each triangle in order of distance to centroid
#     ordered_indices = sortperm(dists)
#     for k in ordered_indices
#         if is_point_in_triangle(x, nodes[elements[k, :], 1:2])
#             return k # return the triangle index
#         end
#     end

#     error("Point $(x) is not inside any triangle.")
# end

# """
#     bool = is_point_in_triangle(x, p)

# Determine if point `x` is in triangle with nodes `p`.
# (See https://stackoverflow.com/a/2049593).
# """
# function is_point_in_triangle(x, p)
#     d₁ = pt_sign(x, p[1, :], p[2, :])
#     d₂ = pt_sign(x, p[2, :], p[3, :])
#     d₃ = pt_sign(x, p[3, :], p[1, :])

#     has_neg = (d₁ < 0) || (d₂ < 0) || (d₃ < 0)
#     has_pos = (d₁ > 0) || (d₂ > 0) || (d₃ > 0)

#     return !(has_neg && has_pos)
# end
# function pt_sign(p₁, p₂, p₃)
#     return (p₁[1] - p₃[1])*(p₂[2] - p₃[2]) - (p₂[1] - p₃[1])*(p₁[2] - p₃[2])
# end