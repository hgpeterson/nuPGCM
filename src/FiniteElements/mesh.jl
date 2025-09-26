struct Mesh{BB, MF, MI, B}
    bbox::BB           # bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
    nodes::MF          # N × 3 matrix of node coordinates
    edges::MI          # ne × 2 matrix of edge node indices
    emap::MI           # nt × n matrix mapping local to global edge indices
    faces::MI          # nf × 3 matrix of face node indices
    fmap::MI           # nt × m matrix mapping local to global face indices
    elements::MI       # M × n matrix of element node indices
    boundary_nodes::B  # map from boundary name to node indices
    boundary_edges::B  # map from boundary name to edge indices
    boundary_faces::B  # map from boundary name to face indices
end

function Mesh(nodes, elements, boundary_nodes)
    bbox = get_bounding_box(nodes)
    emap, edges, boundary_edges = get_edges(elements, boundary_nodes)
    fmap, faces, boundary_faces = get_faces(elements, boundary_nodes)
    return Mesh(bbox, nodes, edges, emap, faces, fmap, elements, boundary_nodes, boundary_edges, boundary_faces)
end

function get_dim(mesh::Mesh)
    return size(mesh.elements, 2) - 1
end

function get_element_type(mesh::Mesh)
    return get_element_type(get_dim(mesh))
end
function get_element_type(connectivities::AbstractMatrix{<:Integer})
    dim = size(connectivities, 2) - 1
    return get_element_type(dim)
end

# function Base.show(io::IO, mesh::Mesh)
#     println(io, "Mesh with dimension $(mesh.dim)")
#     println(io, "├── Nodes: $(size(mesh.nodes, 1))")
#     println(io, "├── Bounding box: x ∈ [$(mesh.bbox[1]), $(mesh.bbox[2])], y ∈ [$(mesh.bbox[3]), $(mesh.bbox[4])], z ∈ [$(mesh.bbox[5]), $(mesh.bbox[6])]")
#     println(io, "├── Interior elements: $(size(mesh.elements, 1))")
#     println(io, "├── Interior element type: $(mesh.element_type)")
#     print(io, "└── Components: $(length(mesh.components))")
# end

# struct MeshComponent{E, MI}
#     name::String
#     element_type::E
#     elements::MI
# end

# function Base.show(io::IO, mc::MeshComponent)
#     println(io, "MeshComponent: $(mc.name)")
#     println(io, "├── Element type: $(mc.element_type)")
#     print(io, "└── Elements: $(size(mc.elements, 1))")
# end

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
        el = Tetrahedron()
        element_type_code = gmsh.model.mesh.get_element_type("Tetrahedron", 1)
    elseif gmsh.model.mesh.get_element_type("Triangle", 1) in element_types
        el = Triangle()
        element_type_code = gmsh.model.mesh.get_element_type("Triangle", 1)
    elseif gmsh.model.mesh.get_element_type("Line", 1) in element_types
        el = Line()
        element_type_code = gmsh.model.mesh.get_element_type("Line", 1)
    else
        throw(ArgumentError("Unsupported element type in mesh."))
    end 
    mesh_dim = dimension(el)
    @info "Mesh dimension = $mesh_dim"

    # get all the interior elements
    _, elements = gmsh.model.mesh.get_elements_by_type(element_type_code)
    elements = Matrix(reshape(Int.(elements), mesh_dim + 1, :)') # M × n matrix

    # get edges and faces of the elements
    edges = gmsh.model.mesh.get_element_edge_nodes(element_type_code)
    # edges = Matrix(reshape(Int.(edges), 2, :)') # ne × 2 matrix
    faces = gmsh.model.mesh.get_element_face_nodes(element_type_code, 3)
    # faces = Matrix(reshape(Int.(faces), 3, :)') # nf × 3 matrix

    # Edge and face tags can then be retrieved by providing their nodes:
    gmsh.model.mesh.create_edges()
    gmsh.model.mesh.create_faces()
    edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(edges)
    faceTags, faceOrientations = gmsh.model.mesh.getFaces(3, faces)

    # Since element edge and face nodes are returned in the same order as the
    # elements, one can easily keep track of which element(s) each edge or face is
    # connected to:
    elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(element_type_code)
    edges2Elements = Dict()
    faces2Elements = Dict()
    for i in 1:length(edgeTags) # 6 edges per tetrahedron
        if !haskey(edges2Elements, edgeTags[i])
            edges2Elements[edgeTags[i]] = [elementTags[div(i - 1, 6) + 1]]
        else
            push!(edges2Elements[edgeTags[i]], elementTags[div(i - 1, 6) + 1])
        end
    end
    for i in 1:length(faceTags) # 4 faces per tetrahedron
        if !haskey(faces2Elements, faceTags[i])
            faces2Elements[faceTags[i]] = [elementTags[div(i - 1, 4) + 1]]
        else
            push!(faces2Elements[faceTags[i]], elementTags[div(i - 1, 4) + 1])
        end
    end

    # # local to global edge mapping
    # edges_l2g = zeros(Int, size(elements))
    # for k in axes(elements, 1)
    #     edge_nodes = gmsh.model.mesh.getElementEdgeNodes(element_type_code, k)
    #     display(Int.(edge_nodes))
    #     edge_tags, _ = gmsh.model.mesh.getEdges(edge_nodes)
    #     display(Int.(edge_tags))
    #     for i in eachindex(edge_tags)
    #         edges_l2g[k, i] = edge_tags[i]
    #     end
    # end

    return edges_l2g

    # # New unique lower dimensional elements can also be easily created given the
    # # edge or face nodes. This is especially useful for numerical methods that
    # # require integrating or interpolating on internal edges or faces (like
    # # e.g. Discontinuous Galerkin techniques), since creating elements for the
    # # internal entities will make this additional mesh data readily available (see
    # # `x6.py'). For example, we can create a new discrete surface...
    # s = gmsh.model.addDiscreteEntity(2)

    # # ... and fill it with unique triangles corresponding to the faces of the
    # # tetrahedra:
    # maxElementTag = gmsh.model.mesh.getMaxElementTag()
    # uniqueFaceTags = Set()
    # tagsForTriangles = []
    # faceNodesForTriangles = []
    # for i in 1:length(faceTags)
    #     if !in(faceTags[i], uniqueFaceTags)
    #         push!(uniqueFaceTags, faceTags[i])
    #         push!(tagsForTriangles, faceTags[i] + maxElementTag)
    #         push!(faceNodesForTriangles,
    #             faceNodes[3 * (i - 1) + 1], faceNodes[3 * (i - 1) + 2], faceNodes[3 * (i - 1) + 3])
    #     end
    # end
    # elementType2D = gmsh.model.mesh.getElementType("triangle", 1)
    # gmsh.model.mesh.addElementsByType(s, elementType2D, tagsForTriangles,
    #                                 faceNodesForTriangles)

    # # Since the tags for the triangles have been created based on the face tags,
    # # the information about neighboring elements can also be readily created,
    # # useful e.g. in Finite Volume or Discontinuous Galerkin techniques:
    # for t in tagsForTriangles
    #     println("triangle " * string(t) * " is connected to tetrahedra " *
    #             string(convert(Array{Int}, faces2Elements[t - maxElementTag])))
    # end

    # # If all you need is the list of all edges or faces in terms of their nodes, you
    # # can also directly call:
    # edgeTags, edgeNodes = gmsh.model.mesh.getAllEdges()
    # faceTags, faceNodes = gmsh.model.mesh.getAllFaces(3)


    # # `physical_groups` is a vector of (`dim`, `tag`) integer pairs
    # physical_groups = gmsh.model.get_physical_groups()
    # # physical_group_names = [gmsh.model.get_physical_name(dim, tag) for (dim, tag) in physical_groups]

    # # initialize
    # components = Vector{MeshComponent}()

    # for (dim, tag) in physical_groups
    #     name = gmsh.model.get_physical_name(dim, tag)
    #     entity_tags = gmsh.model.get_entities_for_physical_group(dim, tag)

    #     element_type = get_element_type(dim)
    #     mc_elements = Matrix{Int}(undef, 0, dim + 1)

    #     # add the elements for each entity in the physical group (component)
    #     for entity_tag in entity_tags
    #         _, _, node_tags = gmsh.model.mesh.get_elements(dim, entity_tag)
    #         @assert length(node_tags) == 1 "Expected exactly one element type for entity $entity."
    #         entity_elements = Matrix(reshape(Int.(node_tags[1]), dim + 1, :)')
    #         mc_elements = vcat(mc_elements, entity_elements)
    #     end

    #     push!(components, MeshComponent(name, element_type, mc_elements))
    # end

    # gmsh.finalize()
    
    # return Mesh(nodes, elements, interior_element_type, components, mesh_dim, bbox)
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

"""
    fmap, faces, boundary_faces = get_faces(elements, boundary_nodes)

1) Find all unique `faces` (nf x 3 matrix) in `elements` (M × n matrix).
2) Determine indices of boundary faces with `boundary_nodes`.
3) Map local faces to global faces with `fmap` (nt x 4 matrix): `fmap[k,i]` is the 
global face number for local face `i` in element `k`.
"""
function get_faces(elements, boundary_nodes)
    if size(elements, 2) != 4
        # no faces for < 3D
        T = eltype(elements)
        return Matrix{T}(undef, 0, 3), Matrix{T}(undef, 0, 2), Dict{String, Vector{T}}()
    end

    # form all faces
    ftag = [elements[:, [1, 2, 3]] 
            elements[:, [1, 2, 4]] 
            elements[:, [1, 3, 4]] 
            elements[:, [2, 3, 4]]]
    nn = 3
    nfaces = size(ftag, 1)

    # sort columns and tag with global indices in last column
    ftag = hcat(sort(ftag, dims=2), 1:nfaces)

    # sort rows
    ftag = sortslices(ftag, dims=1)

    # indices of unique faces
    keep = zeros(Bool, nfaces)
    keep[unique(i -> ftag[i, 1:nn], 1:nfaces)] .= 1

    # keep unique faces
    faces = ftag[keep, 1:nn]

    # boundary edges FIXME: this is slow and will also fail for faces on corners
    T = eltype(elements)
    boundary_faces = Dict{String, Vector{T}}()
    for boundary in keys(boundary_nodes)
        boundary_faces[boundary] = Vector{T}()
        for i in axes(faces, 1)
            if faces[i, 1] in boundary_nodes[boundary] && 
               faces[i, 2] in boundary_nodes[boundary] &&
               faces[i, 3] in boundary_nodes[boundary]
                push!(boundary_faces[boundary], i)
            end
        end
    end

    # mapping from local to global face index
    fmap = cumsum(keep)
    invpermute!(fmap, ftag[:, nn+1])
    fmap = reshape(fmap, :, 4)

    return fmap, faces, boundary_faces
end

"""
    emap, edges, boundary_edges = get_edges(elements, boundary_nodes)

1) Find all unique `edges` (ne x 2 matrix) in `elements` (M × dim+1 matrix).
2) Determine edges on boundary `bndix`.
3) Map local edges to global edges with `emap` (nt x dim+1 matrix): `emap[k,i]` is the 
global edge number for local edge `i` in element `k`.
"""
function get_edges(elements, boundary_nodes)
    T = eltype(elements)

    # dimension of space
    dim = size(elements, 2) - 1

    # get all possible edge index pairs
    ne = T((dim + 1)*dim/2) # number of edges per element = dim + 1 choose 2
    pairs = [1 2
             2 3
             1 3
             1 4
             2 4
             3 4]

    # find all edges
    etag = elements[:, pairs[1, :]]
    for i=2:ne
        etag = vcat(etag, elements[:, pairs[i, :]])
    end
    nedges = size(etag, 1)

    # order node indices so lowest ones are in first column, tag each edge with its global index in 3rd column
    etag = [sort(etag, dims=2)  1:nedges]

    # now sort so that first column goes from lowest to highest node index
    etag = sortslices(etag, dims=1)

    # determine if edge is a duplicate or should be kept
    # keep = zeros(Bool, nedges)
    # keep[unique(i -> etag[i, 1:2], 1:size(etag, 1))] .= 1
    dup = all(etag[2:end, 1:2] .== etag[1:end-1, 1:2], dims=2)[:]
    keep = .![false; dup]

    # only keep unique edges
    edges = etag[keep, 1:2]

    # boundary edges: no duplicates
    dup = [dup; false]
    dup = dup[keep]
    bndix = findall(.!dup)

    # boundary edges FIXME: this is slow and will also fail for edges on corners
    T = eltype(elements)
    boundary_edges = Dict{String, Vector{T}}()
    for boundary in keys(boundary_nodes)
        boundary_edges[boundary] = Vector{T}()
        for i in axes(edges, 1)
            if edges[i, 1] in boundary_nodes[boundary] && edges[i, 2] in boundary_nodes[boundary]
                if i in bndix
                    push!(boundary_edges[boundary], i)
                end
            end
        end
    end

    # compute mapping to global indices
    emap = cumsum(keep)
    invpermute!(emap, etag[:, 3])
    emap = reshape(emap, :, ne)

    return emap, edges, boundary_edges
end

function get_midpoints(mesh::Mesh)
    n_edges = size(mesh.edges, 1)
    midpoints = zeros(eltype(mesh.nodes), n_edges, 3)
    for i in 1:n_edges
        n1, n2 = mesh.edges[i, :]
        midpoints[i, :] = (mesh.nodes[n1, :] + mesh.nodes[n2, :])/2
    end
    return midpoints
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

function Mesh(nx, ny)
    x = range(0, 1; length=nx)
    y = range(0, 1; length=ny)
    nodes = zeros(nx*ny, 3)
    imap = reshape(1:(nx*ny), nx, ny)
    for i in 1:nx, j in 1:ny
        nodes[imap[i, j], :] = [x[i], y[j], 0.0]
    end
    elements = zeros(eltype(nx), 2*(nx-1)*(ny-1), 3)
    k = 1
    for i in 1:(nx-1), j in 1:(ny-1)
        n1 = imap[i, j]
        n2 = imap[i, j+1]
        n3 = imap[i+1, j]
        n4 = imap[i+1, j+1]
        elements[k, :] = [n1, n4, n2]
        k += 1
        elements[k, :] = [n1, n3, n4]
        k += 1
    end
    boundary_nodes = Dict("boundary" => unique(vcat(imap[1, :], imap[end, :], imap[:, 1], imap[:, end])))
    return Mesh(nodes, elements, boundary_nodes)
end