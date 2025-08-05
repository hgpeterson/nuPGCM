struct Mesh{MF, MI, E, VC, TF}
    nodes::MF
    elements::MI
    element_type::E
    components::VC
    dim::Integer
    bbox::TF
end

function Base.show(io::IO, mesh::Mesh)
    println(io, "Mesh with dimension $(mesh.dim)")
    println(io, "├── Nodes: $(size(mesh.nodes, 1))")
    println(io, "├── Bounding box: x ∈ [$(mesh.bbox[1]), $(mesh.bbox[2])], y ∈ [$(mesh.bbox[3]), $(mesh.bbox[4])], z ∈ [$(mesh.bbox[5]), $(mesh.bbox[6])]")
    println(io, "├── Interior elements: $(size(mesh.elements, 1))")
    println(io, "├── Interior element type: $(mesh.element_type)")
    print(io, "└── Components: $(length(mesh.components))")
end

struct MeshComponent{E, MI}
    name::String
    element_type::E
    elements::MI
end

function Base.show(io::IO, mc::MeshComponent)
    println(io, "MeshComponent: $(mc.name)")
    println(io, " ├── Element type: $(mc.element_type)")
    print(io, "└── Elements: $(size(mc.elements, 1))")
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
    _, nodes, _ = gmsh.model.mesh.get_nodes()
    nodes = Matrix(reshape(nodes, 3, :)') # N × 3 matrix

    # bounding box
    bbox = get_bounding_box(nodes)

    # determine the dimension of the mesh
    element_types = gmsh.model.mesh.get_element_types()
    if gmsh.model.mesh.get_element_type("Tetrahedron", 1) in element_types
        interior_element_type = Tetrahedron()
        element_type_code = gmsh.model.mesh.get_element_type("Tetrahedron", 1)
    elseif gmsh.model.mesh.get_element_type("Triangle", 1) in element_types
        interior_element_type = Triangle()
        element_type_code = gmsh.model.mesh.get_element_type("Triangle", 1)
    elseif gmsh.model.mesh.get_element_type("Line", 1) in element_types
        interior_element_type = Line()
        element_type_code = gmsh.model.mesh.get_element_type("Line", 1)
    end 
    mesh_dim = dimension(interior_element_type)
    @info "Mesh dimension = $mesh_dim"

    # get all the interior elements
    _, elements = gmsh.model.mesh.get_elements_by_type(element_type_code)
    elements = Matrix(reshape(Int.(elements), mesh_dim + 1, :)') # M × n matrix

    # `physical_groups` is a vector of (`dim`, `tag`) integer pairs
    physical_groups = gmsh.model.get_physical_groups()
    # physical_group_names = [gmsh.model.get_physical_name(dim, tag) for (dim, tag) in physical_groups]

    # initialize
    components = Vector{MeshComponent}()

    for (dim, tag) in physical_groups
        name = gmsh.model.get_physical_name(dim, tag)
        entity_tags = gmsh.model.get_entities_for_physical_group(dim, tag)

        element_type = get_element_type(dim)
        mc_elements = Matrix{Int}(undef, 0, dim + 1)

        # add the elements for each entity in the physical group (component)
        for entity_tag in entity_tags
            _, _, node_tags = gmsh.model.mesh.get_elements(dim, entity_tag)
            @assert length(node_tags) == 1 "Expected exactly one element type for entity $entity."
            entity_elements = Matrix(reshape(Int.(node_tags[1]), dim + 1, :)')
            mc_elements = vcat(mc_elements, entity_elements)
        end

        push!(components, MeshComponent(name, element_type, mc_elements))
    end

    gmsh.finalize()
    
    return Mesh(nodes, elements, interior_element_type, components, mesh_dim, bbox)
end

function get_bounding_box(nodes)
    xmin = minimum(nodes[:, 1])
    xmax = maximum(nodes[:, 1])
    ymin = minimum(nodes[:, 2])
    ymax = maximum(nodes[:, 2])
    zmin = minimum(nodes[:, 3])
    zmax = maximum(nodes[:, 3])
    return (xmin, xmax, ymin, ymax, zmin, zmax)
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