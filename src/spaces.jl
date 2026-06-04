# Default quadrature order for volume and facet integrals on tetrahedra.
# Order 4 integrates degree-4 polynomials exactly; sufficient for P2 matrix
const QR_ORDER = 4

# Linear geometry interpolation (all meshes are linear tet)
const IP_GEO = Lagrange{RefTetrahedron, 1}()

"""
    cv_u, cv_p, cv_b = make_cell_values(fe_data; qr_order=QR_ORDER)
    cv_u, cv_p, cv_b = make_cell_values(u_order, b_order; qr_order=QR_ORDER)

Return fresh `CellValues` objects for velocity, pressure, and buoyancy assembly.
Create new instances per assembly call; they are mutable (reinit!-ed per cell).
"""
function make_cell_values(u_order::Int, b_order::Int; qr_order=QR_ORDER)
    qr   = QuadratureRule{RefTetrahedron}(qr_order)
    ip_u = Lagrange{RefTetrahedron, u_order}()^3
    ip_p = Lagrange{RefTetrahedron, u_order - 1}()
    ip_b = Lagrange{RefTetrahedron, b_order}()
    return CellValues(qr, ip_u, IP_GEO),
           CellValues(qr, ip_p, IP_GEO),
           CellValues(qr, ip_b, IP_GEO)
end
make_cell_values(fe_data::FEData; kwargs...) =
    make_cell_values(fe_data.u_order, fe_data.b_order; kwargs...)

"""
    fv_u, fv_b = make_facet_values(fe_data; qr_order=QR_ORDER)
    fv_u, fv_b = make_facet_values(u_order, b_order; qr_order=QR_ORDER)

Return fresh `FacetValues` for velocity (wind-stress BC) and buoyancy (surface flux BC).
"""
function make_facet_values(u_order::Int, b_order::Int; qr_order=QR_ORDER)
    fqr  = FacetQuadratureRule{RefTetrahedron}(qr_order)
    ip_u = Lagrange{RefTetrahedron, u_order}()^3
    ip_b = Lagrange{RefTetrahedron, b_order}()
    return FacetValues(fqr, ip_u, IP_GEO),
           FacetValues(fqr, ip_b, IP_GEO)
end
make_facet_values(fe_data::FEData; kwargs...) =
    make_facet_values(fe_data.u_order, fe_data.b_order; kwargs...)

"""
    K = allocate_inversion_matrix(fe_data)

Allocate a sparse matrix for the inversion (Stokes + Coriolis) system using
`dh_up` and `ch_up`. Ferrite's `allocate_matrix(dh, ch)` augments the standard
cell-adjacency sparsity with the entries needed for periodic BC application,
so `apply!` never writes outside the pattern.
"""
allocate_inversion_matrix(fe_data::FEData) =
    allocate_matrix(fe_data.dh_up, fe_data.ch_up)

"""
    M = allocate_evolution_matrix(fe_data)

Allocate a sparse `nb × nb` matrix for the buoyancy evolution system.
"""
allocate_evolution_matrix(fe_data::FEData) = allocate_matrix(fe_data.dh_b, fe_data.ch_b)
