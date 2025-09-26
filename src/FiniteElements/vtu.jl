function save_vtu(mesh::Mesh, filename, data::AbstractDict)
    # points = Matrix(mesh.nodes')
    points = Matrix(vcat(mesh.nodes, get_midpoints(mesh))')
    element_type = get_element_type(mesh)
    # cell_type = get_vtk_cell_type(element_type)
    cell_type = get_vtk_cell_type(element_type, P2())
    dof_data = DoFData(mesh, P2())
    global_dof = dof_data.global_dof
    cells = [MeshCell(cell_type, global_dof[k, :]) for k in axes(global_dof, 1)]
    vtk_grid(filename, points, cells, append=false) do vtk
        for (name, field) in data
            add_field!(vtk, points, cells, name, field)
        end
    end
    @info "VTU file saved to $filename"
end
function add_field!(vtk, points, cells, name, field::AbstractArray)
    vtk[name] = field
end
function add_field!(vtk, points, cells, name, field::FEField)
    vtk[name] = field.values
end
function add_field!(vtk, points, cells, name, field::Function)
    vtk[name] = [field(points[:, i]) for i in axes(points, 2)]
end

# for fun: save a single array as z-coordinate
function save_vtu(mesh::Mesh, filename, data::AbstractArray)
    element_type = get_element_type(mesh)
    if typeof(element_type) != Triangle
        throw(ArgumentError("Only 2D meshes (Triangle elements) supported for this function"))
    end
    # points = Matrix(mesh.nodes')
    points = Matrix(vcat(mesh.nodes, get_midpoints(mesh))')
    points[3, :] = data  # store data in z-coordinate
    # cell_type = get_vtk_cell_type(element_type)
    cell_type = get_vtk_cell_type(element_type, P2())
    # cells = [MeshCell(cell_type, mesh.elements[k, :]) for k in axes(mesh.elements, 1)]
    dof_data = DoFData(mesh, P2())
    global_dof = dof_data.global_dof
    cells = [MeshCell(cell_type, global_dof[k, :]) for k in axes(global_dof, 1)]
    vtk_grid(filename, points, cells, append=false) do vtk
        vtk["z"] = data
    end
    @info "VTU file saved to $filename"
end

get_vtk_cell_type(::Triangle) = VTKCellTypes.VTK_TRIANGLE
get_vtk_cell_type(::Triangle, ::P1) = VTKCellTypes.VTK_TRIANGLE
get_vtk_cell_type(::Triangle, ::P2) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
get_vtk_cell_type(::Tetrahedron) = VTKCellTypes.VTK_TETRA