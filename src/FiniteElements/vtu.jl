function save_vtu(mesh::Mesh, filename, data)
    points = Matrix(mesh.nodes')
    eltype = get_element_type(mesh)
    cell_type = get_vtk_cell_type(eltype)
    cells = [MeshCell(cell_type, mesh.elements[k, :]) for k in axes(mesh.elements, 1)]
    vtk_grid(filename, points, cells, append=false) do vtk
        for (name, values) in data
            vtk[name] = values
        end
    end
    @info "VTU file saved to $filename"
end

get_vtk_cell_type(::Triangle) = VTKCellTypes.VTK_TRIANGLE
get_vtk_cell_type(::Tetrahedron) = VTKCellTypes.VTK_TETRA