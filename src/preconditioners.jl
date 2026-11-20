# Preconditioner for the condensed inversion system
#
#     [ A_uu  A_up ] [u]   [r_u]      A_uu: viscosity + Coriolis
#     [ A_pu   0   ] [p] = [r_p]      A_up: pressure gradient,  A_pu: divergence
#
# It is the upper block-triangular factor of the system, applied as
#
#     p = S̃⁻¹ r_p,    u = Ã_uu⁻¹ (r_u - A_up p),
#
# with approximate inverses of `A_uu` and of the Schur complement
# `S = -A_pu A_uu⁻¹ A_up`.

"""
    CahouetChabardInverse(arch, fe_data, params, forcings, cache, b)

Cahouet–Chabard approximation of the inverse Schur complement,

    S̃⁻¹ = Mν⁻¹ + Kf⁻¹,    Mν = ∫ p q / (2α²ε²ν) dΩ,    Kf = ∫ ∇p·∇q / |f| dΩ.

`Mν⁻¹` is the inverse Schur complement where viscosity dominates; `Kf⁻¹` stands in
for it where rotation does. `Mν` is lumped, and `Kf` is Cholesky factorized on `arch`.
`cache` and `b` are on `arch`.
"""
struct CahouetChabardInverse{V, F, M}
    Mν::V                    # lumped
    Kf::F                    # Cholesky factorization
    rhs::V
    Cᵀ::M                    # maps the pressure DOFs to the free ones
end

function CahouetChabardInverse(arch::AbstractArchitecture, fe_data::FEData,
                               params::Parameters, forcings::Forcings, cache, b::AbstractVector)
    M, Kf = build_pressure_matrices(fe_data, params)
    # Kf is singular for constant pressure; a small multiple of M makes it positive definite
    Kf += 1e-8 * tr(Kf) / tr(M) * M
    Cᵀ = on_architecture(arch, sparse(fe_data.C_up[fe_data.p_dof_indices, fe_data.free_p]'))
    return CahouetChabardInverse(viscous_schur_complement(cache, Cᵀ, params, forcings, b),
                                 cholesky(Symmetric(on_architecture(arch, Kf))),
                                 on_architecture(arch, zeros(size(M, 1))), Cᵀ)
end

function LinearAlgebra.mul!(y, S::CahouetChabardInverse, x)
    S.rhs .= x
    y .= x ./ S.Mν .+ S.Kf \ S.rhs
    return y
end

"""
    M, Kf = build_pressure_matrices(fe_data, params)

Assemble, on the free pressure DOFs,

    M = ∫ p q dΩ,    Kf = ∫ ∇p·∇q / |f| dΩ.
"""
function build_pressure_matrices(fe_data::FEData, params::Parameters)
    dh_up   = fe_data.dh_up
    _, cv_p, _ = make_cell_values(fe_data)
    n_p     = getnbasefunctions(cv_p)
    p_range = dof_range(dh_up, :p)
    f_min   = 1e-2   # floor on |f|, which vanishes at the equator

    rows = Int[]; cols = Int[]
    M_vals = Float64[]; Kf_vals = Float64[]
    M_cell  = zeros(n_p, n_p)
    Kf_cell = zeros(n_p, n_p)

    for cell in CellIterator(dh_up)
        reinit!(cv_p, cell)
        coords = getcoordinates(cell)
        fill!(M_cell, 0.0); fill!(Kf_cell, 0.0)

        for q in 1:getnquadpoints(cv_p)
            x  = spatial_coordinate(cv_p, q, coords)
            dΩ = getdetJdV(cv_p, q)
            abs_f = max(abs(params.f(x)), f_min)

            for i in 1:n_p, j in 1:n_p
                φᵢφⱼ   = shape_value(cv_p, q, i) * shape_value(cv_p, q, j)
                ∇φᵢ∇φⱼ = shape_gradient(cv_p, q, i) ⋅ shape_gradient(cv_p, q, j)
                M_cell[i, j]  += φᵢφⱼ * dΩ
                Kf_cell[i, j] += ∇φᵢ∇φⱼ / abs_f * dΩ
            end
        end

        dofs = celldofs(cell)[p_range]
        for (j, col) in enumerate(dofs), (i, row) in enumerate(dofs)
            push!(rows, row); push!(cols, col)
            push!(M_vals, M_cell[i, j])
            push!(Kf_vals, Kf_cell[i, j])
        end
    end

    N = ndofs(dh_up)
    C = fe_data.C_up[:, fe_data.free_p]
    condense(vals) = C' * sparse(rows, cols, vals, N, N) * C
    return condense(M_vals), condense(Kf_vals)
end

"""
    Mν = viscous_schur_complement(cache, Cᵀ, params, forcings, b)

Lumped `Mν = ∫ p q / (2α²ε²ν) dΩ` on the free pressure DOFs, the Schur complement
where viscosity dominates, on the architecture of `cache`. `ν` is the eddy viscosity for
the buoyancy `b` when the eddy parameterization is on, and `forcings.ν` otherwise. `Cᵀ`
maps the pressure DOFs to the free ones.
"""
function viscous_schur_complement(cache::AssemblyCache, Cᵀ, params::Parameters,
                                  forcings::Forcings, b::AbstractVector)
    Mν = similar(b, size(Cᵀ, 2))
    fill!(Mν, 0)
    viscous_schur_kernel!(get_backend(Mν))(Mν, b, cache, forcings.ν, forcings.eddy_param,
                                           params.α, params.ε, params.N²; ndrange=size(cache.dofs_p))
    return Cᵀ * Mν
end

@kernel function viscous_schur_kernel!(Mν, b, cache, ν⁰, eddy_param, α, ε, N²)
    i, c = @index(Global, NTuple)
    mᵢ = 0.0
    for q in axes(cache.phi_p, 1)
        x = cache.x0[c] + cache.J[c] ⋅ cache.x_ref[q]
        if eddy_param.is_on
            ν = ν_eddy(eddy_param, eddy_param.f(x), α * (N² + vertical_derivative(b, cache, q, c)))
        else
            ν = ν⁰ isa Function ? ν⁰(x) : ν⁰
        end
        # row sum of φᵢφⱼ is φᵢ
        mᵢ += cache.phi_p[q, i] / (2α^2 * ε^2 * ν) * cache.w[q] * cache.detJ[c]
    end
    @atomic Mν[cache.dofs_p[i, c]] += mᵢ
end

"""
    Ã = ILU0(A)

ILU(0) factorization of the sparse GPU matrix `A`, applied as `ldiv!(y, Ã, x)` by two
triangular solves whose analysis is done here, once.
"""
struct ILU0{F, T, V}
    factors::F               # L (unit diagonal) and U in one matrix
    L::T
    U::T
    z::V
end

function ILU0(A)
    factors = KrylovPreconditioners.kp_ilu0(A)
    return ILU0(factors, KrylovPreconditioners.TriangularOperator(factors.P, 'L', 'U'),
                KrylovPreconditioners.TriangularOperator(factors.P, 'U', 'N'),
                similar(nonzeros(A), size(A, 1)))
end

function LinearAlgebra.ldiv!(y, Ã::ILU0, x)
    ldiv!(Ã.z, Ã.L, x)
    ldiv!(y, Ã.U, Ã.z)
    return y
end

function update_P!(P::ILU0, A, diagonal=nothing)
    KrylovPreconditioners.update!(P.factors, A)
    KrylovPreconditioners.update!(P.L, P.factors.P)
    KrylovPreconditioners.update!(P.U, P.factors.P)
    return P
end

"""
    P = BlockTriangularPreconditioner(arch, fe_data, A, schur_inverse)

Block-triangular preconditioner for the condensed inversion matrix `A`, applied as
`mul!(y, P, x)`. The velocity block `A_uu` is factorized by ILU(0) on the GPU and by a
sparse LU on the CPU; `schur_inverse` approximates the inverse Schur complement.
`nzidx` is the position in `nonzeros(A)` of each entry of `nonzeros(A_uu)`.
"""
struct BlockTriangularPreconditioner{M, F, I, S, V}
    A_uu::M
    Ã_uu::F                  # factorization of A_uu
    nzidx::I
    schur_inverse::S
    A_up::M
    free_u::UnitRange{Int}
    free_p::UnitRange{Int}
    u_rhs::V
end

function BlockTriangularPreconditioner(arch::AbstractArchitecture, fe_data::FEData,
                                       A, schur_inverse)
    free_u, free_p = fe_data.free_u, fe_data.free_p
    A_cpu = on_architecture(CPU(), A)
    A_uu  = A_cpu[free_u, free_u]
    # free_u leads free_dofs, so entry (i, j) of A_uu is entry (i, j) of A
    csc_index = [first(nzrange(A_cpu, j)) - 1 +
                 searchsortedfirst(view(rowvals(A_cpu), nzrange(A_cpu, j)), i)
                 for j in axes(A_uu, 2) for i in view(rowvals(A_uu), nzrange(A_uu, j))]
    A_uu  = on_architecture(arch, A_uu)
    nzidx = nonzero_positions(A)[csc_index][invperm(nonzero_positions(A_uu))]
    Ã_uu  = arch isa GPU ? ILU0(A_uu) : lu(A_uu)
    return BlockTriangularPreconditioner(A_uu, Ã_uu, on_architecture(arch, Int32.(nzidx)),
                                         schur_inverse, on_architecture(arch, A_cpu[free_u, free_p]),
                                         free_u, free_p, on_architecture(arch, zeros(length(free_u))))
end

function LinearAlgebra.mul!(y, P::BlockTriangularPreconditioner, x)
    x_u, x_p = view(x, P.free_u), view(x, P.free_p)
    y_u, y_p = view(y, P.free_u), view(y, P.free_p)
    mul!(y_p, P.schur_inverse, x_p)
    P.u_rhs .= x_u
    mul!(P.u_rhs, P.A_up, y_p, -1.0, 1.0)
    ldiv!(y_u, P.Ã_uu, P.u_rhs)
    return y
end
