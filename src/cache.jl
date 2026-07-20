"""
    AssemblyCache

Precomputed, mesh-constant data for fast re-assembly at each timestep (advection
RHS, convection-parameterization rebuilds). All meshes use linear tetrahedra, so
each cell's Jacobian is constant: physical gradients are `J⁻ᵀ ∇̂φ` with per-cell
`J⁻ᵀ` (see [`_∂ᵣ`](@ref)), and `dΩ_q = w_q |det J|`. Shape functions are
tabulated once in reference space; the kernels then run on flat arrays with no
`reinit!` and no per-cell allocation.

Fields:
- `w`: quadrature weights (length nq)
- `phi_b[q, i]`, `dphi_b[:, i, q]`: buoyancy basis values / reference gradients
- `phi_u[q, j]`: scalar velocity basis values
- `Jinv_t[c]`, `detJ[c]`: per-cell inverse-transpose Jacobian and |det J|
- `dofs_b[:, c]`: global b DOFs of cell c
- `dofs_u[:, c]`: global u DOFs of cell c within `dh_up` (interleaved xyz)
- `nzidx_b[n_b*(j-1)+i, c]`: linear index into `nzval` of an evolution-pattern
  matrix for local entry (i, j) of cell c (column-major local numbering)
"""
struct AssemblyCache
    w::Vector{Float64}
    phi_b::Matrix{Float64}
    dphi_b::Array{Float64, 3}
    phi_u::Matrix{Float64}
    Jinv_t::Vector{Tensor{2, 3, Float64, 9}}
    detJ::Vector{Float64}
    dofs_b::Matrix{Int}
    dofs_u::Matrix{Int}
    nzidx_b::Matrix{Int}
end

"""
    _∂ᵣ(Jᵀ, r, g1, g2, g3)

Component `r` of the physical gradient `J⁻ᵀ ĝ` from reference-gradient
components `(g1, g2, g3)`, with `Jᵀ = J⁻ᵀ` from the cache.
"""
@inline _∂ᵣ(Jᵀ, r, g1, g2, g3) = Jᵀ[r, 1]*g1 + Jᵀ[r, 2]*g2 + Jᵀ[r, 3]*g3

function Base.summary(cache::AssemblyCache)
    t = typeof(cache)
    return "$(parentmodule(t)).$(nameof(t))"
end

function AssemblyCache(dh_up::DofHandler, dh_b::DofHandler, K_b::SparseMatrixCSC,
                       u_order::Int, b_order::Int)
    grid = Ferrite.get_grid(dh_b)
    qr   = QuadratureRule{RefTetrahedron}(QR_ORDER)
    ip_b = Lagrange{RefTetrahedron, b_order}()
    ip_u = Lagrange{RefTetrahedron, u_order}()   # scalar; vector dofs interleave xyz
    nq   = getnquadpoints(qr)
    n_b  = getnbasefunctions(ip_b)
    n_su = getnbasefunctions(ip_u)
    ncells = getncells(grid)
    u_range = dof_range(dh_up, :u)

    # reference-space tables
    w      = copy(Ferrite.getweights(qr))
    phi_b  = zeros(nq, n_b)
    dphi_b = zeros(3, n_b, nq)
    phi_u  = zeros(nq, n_su)
    for (q, ξ) in enumerate(Ferrite.getpoints(qr))
        for i in 1:n_b
            phi_b[q, i]      = Ferrite.reference_shape_value(ip_b, ξ, i)
            dphi_b[:, i, q] .= Ferrite.reference_shape_gradient(ip_b, ξ, i)
        end
        for j in 1:n_su
            phi_u[q, j] = Ferrite.reference_shape_value(ip_u, ξ, j)
        end
    end

    # per-cell geometry, DOF maps, and nzval index map into the evolution pattern
    Jinv_t  = Vector{Tensor{2, 3, Float64, 9}}(undef, ncells)
    detJ    = zeros(ncells)
    dofs_b  = zeros(Int, n_b, ncells)
    dofs_u  = zeros(Int, 3*n_su, ncells)
    nzidx_b = zeros(Int, n_b*n_b, ncells)
    rowval  = K_b.rowval
    colptr  = K_b.colptr
    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        c = cellid(cc_b)
        coords = getcoordinates(cc_b)
        J = Tensor{2, 3}(hcat(coords[2] - coords[1],
                              coords[3] - coords[1],
                              coords[4] - coords[1]))
        Jinv_t[c] = transpose(inv(J))
        detJ[c]   = abs(det(J))
        dofs_b[:, c] .= celldofs(cc_b)
        dofs_u[:, c] .= @view celldofs(cc_up)[u_range]
        for j in 1:n_b
            gj = dofs_b[j, c]
            rng = colptr[gj]:(colptr[gj + 1] - 1)
            for i in 1:n_b
                gi = dofs_b[i, c]
                p  = searchsortedfirst(rowval, gi, first(rng), last(rng), Base.Order.Forward)
                @assert p <= last(rng) && rowval[p] == gi "entry ($gi, $gj) not in pattern"
                nzidx_b[n_b*(j - 1) + i, c] = p
            end
        end
    end

    return AssemblyCache(w, phi_b, dphi_b, phi_u,
                         Jinv_t, detJ, dofs_b, dofs_u, nzidx_b)
end
