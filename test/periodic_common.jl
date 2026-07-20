# Shared fixtures for the periodic-box test files
# (test_periodic_box.jl, test_periodic_blob.jl, test_periodic_advection.jl).

"""
    box_file = ensure_periodic_box_mesh(h, α; W=1.0, L=1.0)

Return the periodic-box mesh path, generating the mesh if it does not exist.
"""
function ensure_periodic_box_mesh(h, α; W=1.0, L=1.0)
    box_file = joinpath(@__DIR__, @sprintf("../meshes/periodic_box_h%.2e_a%.2e.msh", h, α))
    if !isfile(box_file)
        include(joinpath(@__DIR__, "../meshes/periodic_box.jl"))
        Base.invokelatest(mesh_periodic_box, h, α; W, L)
    end
    return box_file
end

"""
    x_up = fill_u(fe_data, fn)
    b    = fill_b(fe_data, fn)

DOF vectors holding the nodal interpolant of an analytic field: `fn(x)::Vec{3}`
for velocity (length `ndofs(dh_up)`, p entries zero) or `fn(x)::Real` for
buoyancy (length `nb`).
"""
function fill_u(fe_data, fn)
    x = zeros(ndofs(fe_data.dh_up))
    apply_analytical!(x, fe_data.dh_up, :u, fn)
    return x
end
function fill_b(fe_data, fn)
    b = zeros(fe_data.nb)
    apply_analytical!(b, fe_data.dh_b, :b, fn)
    return b
end
