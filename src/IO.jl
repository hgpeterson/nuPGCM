function save_state(model::Model, ofile)
    s = model.state
    t = isnothing(model.timestepper) ? 0.0 : model.timestepper.t[]
    jldsave(ofile; u=s.u, p=s.p, b=s.b, t=t)
    @info "Model state saved to '$ofile'"
end

function set_state_from_file!(m::Model, ifile)
    d = jldopen(ifile, "r")
    m.state.u .= d["u"]
    m.state.p .= d["p"]
    m.state.b .= d["b"]
    if !isnothing(m.timestepper)
        m.timestepper.t[] = d["t"]
    end
    close(d)
    @info "Model state set from '$ifile'"
    return m
end

"""
    save_vtk_p2(model; ofile, kwargs...)

Write the model state to `ofile.vtu` on quadratic tetrahedra, which represent the
P2 fields exactly. `kwargs` are passed to `WriteVTK.vtk_grid`.
"""
function save_vtk_p2(m::Model; ofile="$out_dir/data/state", vtk_kwargs...)
    fe_data  = m.fe_data
    state    = m.state
    dh_up    = fe_data.dh_up
    dh_b     = fe_data.dh_b
    nb       = fe_data.nb

    # "quadrature" points at the P2 nodes, to evaluate fields and coordinates there
    ip_b    = Lagrange{RefTetrahedron, fe_data.b_order}()
    ip_u    = Lagrange{RefTetrahedron, fe_data.u_order}()^3
    ip_p    = Lagrange{RefTetrahedron, fe_data.u_order - 1}()
    ref_pts = Ferrite.reference_coordinates(ip_b)
    qr      = QuadratureRule{RefTetrahedron}(zeros(length(ref_pts)), ref_pts)
    cv_b = CellValues(qr, ip_b, IP_GEO)
    cv_u = CellValues(qr, ip_u, IP_GEO)
    cv_p = CellValues(qr, ip_p, IP_GEO)
    nq   = getnquadpoints(cv_b)

    u_range = dof_range(dh_up, :u)
    p_range = dof_range(dh_up, :p)

    x_up = zeros(ndofs(dh_up))
    x_up[fe_data.u_dof_indices] .= state.u
    x_up[fe_data.p_dof_indices] .= state.p

    # indexed by dh_b DOF, i.e. by P2 node
    coords_p2 = zeros(3, nb)
    u_p2      = zeros(3, nb)
    p_p2      = zeros(nb)

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_b, cc_b)
        reinit!(cv_u, cc_up)
        reinit!(cv_p, cc_up)
        cell_coords = getcoordinates(cc_b)
        dofs_b  = celldofs(cc_b)
        dofs_up = celldofs(cc_up)
        ue = x_up[dofs_up[u_range]]
        pe = x_up[dofs_up[p_range]]
        for q in 1:nq
            dof = dofs_b[q]
            x   = spatial_coordinate(cv_b, q, cell_coords)
            coords_p2[:, dof] .= x
            u_p2[:, dof]      .= function_value(cv_u, q, ue)
            p_p2[dof]          = function_value(cv_p, q, pe)
        end
    end

    # celldofs returns a buffer that the iterator reuses, hence the copy
    cells = [WriteVTK.MeshCell(VTKCellTypes.VTK_QUADRATIC_TETRA, copy(celldofs(cc)))
             for cc in CellIterator(dh_b)]

    b_full = state.b .+ m.params.N² .* coords_p2[3, :]

    vtk = WriteVTK.vtk_grid(ofile, coords_p2, cells; append=false, vtk_kwargs...)
    WriteVTK.vtk_point_data(vtk, u_p2,   "u")
    WriteVTK.vtk_point_data(vtk, p_p2,   "p")
    WriteVTK.vtk_point_data(vtk, b_full, "b")
    WriteVTK.vtk_save(vtk)
    @info "VTK P2 state saved to '$ofile.vtu'"
    return nothing
end

"""
    save_vtk(model; ofile, kwargs...)

Write the model state to `ofile.vtu` on linear tetrahedra, i.e. with the fields
sampled at the cell vertices. `kwargs` are passed to `VTKGridFile`.
"""
function save_vtk(m::Model; ofile="$out_dir/data/state", vtk_kwargs...)
    fe_data = m.fe_data
    grid    = fe_data.mesh.grid
    state   = m.state

    x_up = zeros(ndofs(fe_data.dh_up))
    x_up[fe_data.u_dof_indices] .= state.u
    x_up[fe_data.p_dof_indices] .= state.p

    b_full = state.b .+ m.params.N² .* b_dof_z(fe_data)

    VTKGridFile(ofile, grid; append=false, vtk_kwargs...) do vtk
        write_solution(vtk, fe_data.dh_up, x_up)
        write_solution(vtk, fe_data.dh_b,  b_full)
    end
    @info "VTK state saved to '$ofile.vtu'"
end
