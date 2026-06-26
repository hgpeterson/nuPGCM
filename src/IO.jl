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
    save_vtk_p2(model; ofile="...", kwargs...)

Write the current model state to a VTK file using `VTK_QUADRATIC_TETRA` cells
(cell type 24, 10 nodes) so that the P2 velocity and buoyancy fields are
represented exactly in ParaView, rather than being evaluated at the 4 P1 corner
nodes only.

## Why this is needed

Ferrite's `VTKGridFile` infers the VTK cell type from the *grid cell type*:
`Tetrahedron` → `VTK_TETRA` (4 nodes), `QuadraticTetrahedron` → `VTK_QUADRATIC_TETRA`
(10 nodes). Our mesh uses `Tetrahedron` (P1 geometry, `IP_GEO = Lagrange 1`) even
though the solution fields are P2. `save_vtk` therefore evaluates the P2 solution
at the 4 corner nodes and writes a linear representation, losing the quadratic
variation within each cell.

`save_vtk_p2` works around this by manually building the 10-node P2 point
coordinates and connectivity directly from the `dh_b` DOF structure, bypassing
the grid cell type entirely.

## Node ordering

Ferrite's `reference_coordinates(Lagrange{RefTetrahedron, 2}())` lists nodes in
the same order as VTK's `VTK_QUADRATIC_TETRA`: corners 1–4 first, then the 6
edge midpoints in the order (1,2),(2,3),(1,3),(1,4),(2,4),(3,4). No permutation
is required.

## Pitfall: celldofs shares a buffer

`celldofs(cc)` returns a *view* of a reused internal buffer. The connectivity
array for each `MeshCell` must be `copy`ed immediately, or all cells will alias
the last cell's DOFs after the iterator advances.
"""
function save_vtk_p2(m::Model; ofile="$out_dir/data/state", vtk_kwargs...)
    fe_data  = m.fe_data
    state    = m.state
    dh_up    = fe_data.dh_up
    dh_b     = fe_data.dh_b
    nb       = fe_data.nb

    # quadrature points placed at the P2 Lagrange reference coordinates so that
    # spatial_coordinate(cv_b, q, coords) gives each P2 node's physical position
    # and function_value(cv_u/p, q, ue) gives the exact DOF value at that node
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

    # build arrays indexed by dh_b DOF number (= P2 node index, 1:nb)
    coords_p2 = zeros(3, nb)
    u_p2      = zeros(3, nb)
    p_p2      = zeros(nb)

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_b, cc_b)
        reinit!(cv_u, cc_up)
        reinit!(cv_p, cc_up)
        cell_coords = getcoordinates(cc_b)
        dofs_b  = celldofs(cc_b)   # view — used only within this iteration
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

    # copy(celldofs(cc)): celldofs returns a view of a shared buffer; without
    # the copy every MeshCell would alias the last cell after the loop ends.
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
    save_vtk(model; ofile="...")

Write the current model state to a VTK file using Ferrite's `VTKGridFile`.
Fields are evaluated at the P1 corner nodes only (linear representation).
Use `save_vtk_p2` to capture the full quadratic variation.
The file name is `ofile.vtu`.
"""
function save_vtk(m::Model; ofile="$out_dir/data/state", vtk_kwargs...)
    fe_data = m.fe_data
    grid    = fe_data.mesh.grid
    state   = m.state

    # reassemble the combined (u, p) DOF vector in dh_up ordering
    x_up = zeros(ndofs(fe_data.dh_up))
    x_up[fe_data.u_dof_indices] .= state.u
    x_up[fe_data.p_dof_indices] .= state.p

    # compute z-coordinate of each b DOF via P2 reference coordinates
    ip_b    = Lagrange{RefTetrahedron, fe_data.b_order}()
    ref_pts = Ferrite.reference_coordinates(ip_b)
    qr_pt   = QuadratureRule{RefTetrahedron}(zeros(length(ref_pts)), ref_pts)
    cv_b    = CellValues(qr_pt, ip_b, IP_GEO)
    z_b     = zeros(fe_data.nb)
    for cc in CellIterator(fe_data.dh_b)
        reinit!(cv_b, cc)
        coords = getcoordinates(cc)
        dofs   = celldofs(cc)
        for q in 1:getnquadpoints(cv_b)
            z_b[dofs[q]] = spatial_coordinate(cv_b, q, coords)[3]
        end
    end
    b_full = state.b .+ m.params.N² .* z_b

    VTKGridFile(ofile, grid; append=false, vtk_kwargs...) do vtk
        write_solution(vtk, fe_data.dh_up, x_up)
        write_solution(vtk, fe_data.dh_b,  b_full)
    end
    @info "VTK state saved to '$ofile.vtu'"
end
