"""
    AssemblyCache(dh_up, dh_b, u_dof_indices, p_dof_indices, u_order, b_order)

Reference shape functions and cell geometry for assembly loops that run on flat
arrays instead of Ferrite's `CellValues`. Cells are linear tetrahedra, so the
Jacobian is constant on each. `on_architecture(arch, cache)` moves it to `arch`.

- `w[q]`, `x_ref[q]`: quadrature weights and reference points
- `phi_b[q, i]`, `dphi_b[:, i, q]`: buoyancy basis values and reference gradients
- `phi_u[q, j]`, `dphi_u[:, j, q]`: the same for the scalar velocity basis; local
  velocity DOF `3(j-1)+k` is component `k` of node `j`
- `phi_p[q, i]`: pressure basis values
- `x0[c]`, `J[c]`, `Jinv_t[c]`, `detJ[c]`: first vertex, Jacobian, its inverse
  transpose and `|det J|` of cell `c`
- `dofs_b[:, c]`, `dofs_u[:, c]`, `dofs_p[:, c]`: global DOFs of cell `c` in `dh_b`, and
  the positions of its velocity and pressure DOFs in the velocity and pressure vectors,
  which are ordered by `u_dof_indices` and `p_dof_indices`
"""
struct AssemblyCache{V, X, M, A, T, I}
    w::V
    x_ref::X
    phi_b::M
    dphi_b::A
    phi_u::M
    dphi_u::A
    phi_p::M
    x0::X
    J::T
    Jinv_t::T
    detJ::V
    dofs_b::I
    dofs_u::I
    dofs_p::I
end

Adapt.@adapt_structure AssemblyCache

on_architecture(arch::AbstractArchitecture, cache::AssemblyCache) =
    AssemblyCache(map(name -> on_architecture(arch, getfield(cache, name)), fieldnames(AssemblyCache))...)

"""
    _∂ᵣ(J⁻ᵀ, r, g1, g2, g3)

Component `r` of the physical gradient `J⁻ᵀ ĝ` of the reference gradient
`ĝ = (g1, g2, g3)`.
"""
@inline _∂ᵣ(J⁻ᵀ, r, g1, g2, g3) = J⁻ᵀ[r, 1]*g1 + J⁻ᵀ[r, 2]*g2 + J⁻ᵀ[r, 3]*g3

"""
    ∂z_b = vertical_derivative(b, cache, q, c)

`∂z b` at quadrature point `q` of cell `c`, for the buoyancy DOF vector `b`.
"""
@inline function vertical_derivative(b, cache, q, c)
    g1 = 0.0; g2 = 0.0; g3 = 0.0
    for i in axes(cache.dofs_b, 1)
        bᵢ = b[cache.dofs_b[i, c]]
        g1 += cache.dphi_b[1, i, q] * bᵢ
        g2 += cache.dphi_b[2, i, q] * bᵢ
        g3 += cache.dphi_b[3, i, q] * bᵢ
    end
    return _∂ᵣ(cache.Jinv_t[c], 3, g1, g2, g3)
end

function Base.summary(cache::AssemblyCache)
    t = typeof(cache)
    return "$(parentmodule(t)).$(nameof(t))"
end

function AssemblyCache(dh_up::DofHandler, dh_b::DofHandler, u_dof_indices::Vector{Int},
                       p_dof_indices::Vector{Int}, u_order::Int, b_order::Int)
    grid = Ferrite.get_grid(dh_b)
    qr   = QuadratureRule{RefTetrahedron}(QR_ORDER)
    ip_b = Lagrange{RefTetrahedron, b_order}()
    ip_u = Lagrange{RefTetrahedron, u_order}()
    ip_p = Lagrange{RefTetrahedron, u_order - 1}()
    nq   = getnquadpoints(qr)
    n_b  = getnbasefunctions(ip_b)
    n_su = getnbasefunctions(ip_u)
    n_p  = getnbasefunctions(ip_p)
    ncells = getncells(grid)
    u_range = dof_range(dh_up, :u)
    p_range = dof_range(dh_up, :p)
    position = zeros(Int, ndofs(dh_up))   # of each dh_up DOF in the velocity or pressure vector
    position[u_dof_indices] .= eachindex(u_dof_indices)
    position[p_dof_indices] .= eachindex(p_dof_indices)

    # reference-space tables
    w      = copy(Ferrite.getweights(qr))
    x_ref  = copy(Ferrite.getpoints(qr))
    phi_b  = zeros(nq, n_b)
    dphi_b = zeros(3, n_b, nq)
    phi_u  = zeros(nq, n_su)
    dphi_u = zeros(3, n_su, nq)
    phi_p  = zeros(nq, n_p)
    for (q, ξ) in enumerate(Ferrite.getpoints(qr))
        for i in 1:n_b
            phi_b[q, i]      = Ferrite.reference_shape_value(ip_b, ξ, i)
            dphi_b[:, i, q] .= Ferrite.reference_shape_gradient(ip_b, ξ, i)
        end
        for j in 1:n_su
            phi_u[q, j]      = Ferrite.reference_shape_value(ip_u, ξ, j)
            dphi_u[:, j, q] .= Ferrite.reference_shape_gradient(ip_u, ξ, j)
        end
        for i in 1:n_p
            phi_p[q, i] = Ferrite.reference_shape_value(ip_p, ξ, i)
        end
    end

    # per-cell geometry and DOF maps
    x0     = Vector{Vec{3, Float64}}(undef, ncells)
    Jcache = Vector{Tensor{2, 3, Float64, 9}}(undef, ncells)
    Jinv_t = Vector{Tensor{2, 3, Float64, 9}}(undef, ncells)
    detJ   = zeros(ncells)
    dofs_b = zeros(Int, n_b, ncells)
    dofs_u = zeros(Int, 3*n_su, ncells)
    dofs_p = zeros(Int, n_p, ncells)
    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        c = cellid(cc_b)
        coords = getcoordinates(cc_b)
        J = Tensor{2, 3}(hcat(coords[2] - coords[1],
                              coords[3] - coords[1],
                              coords[4] - coords[1]))
        x0[c]     = coords[1]
        Jcache[c] = J
        Jinv_t[c] = transpose(inv(J))
        detJ[c]   = abs(det(J))
        dofs_b[:, c] .= celldofs(cc_b)
        dofs_u[:, c] .= @view position[celldofs(cc_up)[u_range]]
        dofs_p[:, c] .= @view position[celldofs(cc_up)[p_range]]
    end

    return AssemblyCache(w, x_ref, phi_b, dphi_b, phi_u, dphi_u, phi_p,
                         x0, Jcache, Jinv_t, detJ, dofs_b, dofs_u, dofs_p)
end

"""
    nzidx = build_nzidx_map(K, dofs)

Index into `K.nzval` of each cell's local block: `nzidx[n*(j-1)+i, c]` is the
index of entry `(dofs[i, c], dofs[j, c])`, with `n = size(dofs, 1)`, or 0 if either
DOF is 0.
"""
function build_nzidx_map(K::SparseMatrixCSC, dofs::Matrix{Int})
    n, ncells = size(dofs)
    nzidx  = zeros(Int, n*n, ncells)
    rowval = K.rowval
    colptr = K.colptr
    for c in 1:ncells
        for j in 1:n
            gj  = dofs[j, c]
            gj == 0 && continue
            rng = colptr[gj]:(colptr[gj + 1] - 1)
            for i in 1:n
                gi = dofs[i, c]
                gi == 0 && continue
                nzidx[n*(j - 1) + i, c] = searchsortedfirst(rowval, gi, first(rng), last(rng),
                                                            Base.Order.Forward)
            end
        end
    end
    return nzidx
end
