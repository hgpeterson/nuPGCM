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
    CahouetChabardInverse(arch, fe_data, params, forcings, b)

Cahouet–Chabard approximation of the inverse Schur complement,

    S̃⁻¹ = Mν⁻¹ + Kf⁻¹,    Mν = ∫ p q / (2α²ε²ν) dΩ,    Kf = ∫ ∇p·∇q / |f| dΩ.

`Mν⁻¹` is the inverse Schur complement where viscosity dominates; `Kf⁻¹` stands in
for it where rotation does. `Mν` is lumped, and `Kf` is factorized on the CPU.
"""
struct CahouetChabardInverse{V, F}
    Mν_inverse::V            # 1 / (lumped Mν)
    Kf::F                    # Cholesky factorization
    x_cpu::Vector{Float64}
    Kf_solution::V
end

function CahouetChabardInverse(arch::AbstractArchitecture, fe_data::FEData,
                               params::Parameters, forcings::Forcings, b::AbstractVector)
    M, Mν, Kf = build_pressure_matrices(fe_data, params, forcings, b)
    # Kf is singular for constant pressure; a small multiple of M makes it positive definite
    Kf += 1e-8 * tr(Kf) / tr(M) * M
    n = size(M, 1)
    return CahouetChabardInverse(on_architecture(arch, 1 ./ vec(sum(Mν, dims=2))),
                                 cholesky(Symmetric(Kf)),
                                 zeros(n), on_architecture(arch, zeros(n)))
end

function LinearAlgebra.mul!(y, S::CahouetChabardInverse, x)
    copyto!(S.x_cpu, x)
    copyto!(S.Kf_solution, S.Kf \ S.x_cpu)
    y .= S.Mν_inverse .* x .+ S.Kf_solution
    return y
end

"""
    M, Mν, Kf = build_pressure_matrices(fe_data, params, forcings, b)

Assemble, on the free pressure DOFs,

    M = ∫ p q dΩ,    Mν = ∫ p q / (2α²ε²ν) dΩ,    Kf = ∫ ∇p·∇q / |f| dΩ,

where `ν` is the eddy viscosity for the buoyancy `b` when the eddy parameterization
is on, and `forcings.ν` otherwise.
"""
function build_pressure_matrices(fe_data::FEData, params::Parameters, forcings::Forcings,
                                 b::AbstractVector)
    dh_up, dh_b = fe_data.dh_up, fe_data.dh_b
    _, cv_p, cv_b = make_cell_values(fe_data)
    n_p     = getnbasefunctions(cv_p)
    p_range = dof_range(dh_up, :p)
    α²ε²    = params.α^2 * params.ε^2
    α, N²   = params.α, params.N²
    eddy_param = forcings.eddy_param
    f_min   = 1e-2   # floor on |f|, which vanishes at the equator

    rows = Int[]; cols = Int[]
    M_vals = Float64[]; Mν_vals = Float64[]; Kf_vals = Float64[]
    M_cell  = zeros(n_p, n_p)
    Mν_cell = zeros(n_p, n_p)
    Kf_cell = zeros(n_p, n_p)

    for (cell_up, cell_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_p, cell_up)
        reinit!(cv_b, cell_b)
        coords = getcoordinates(cell_up)
        b_cell = b[celldofs(cell_b)]
        fill!(M_cell, 0.0); fill!(Mν_cell, 0.0); fill!(Kf_cell, 0.0)

        for q in 1:getnquadpoints(cv_p)
            x  = spatial_coordinate(cv_p, q, coords)
            dΩ = getdetJdV(cv_p, q)
            if eddy_param.is_on
                ∂z_b = function_gradient(cv_b, q, b_cell)[3]
                ν = ν_eddy(eddy_param, eddy_param.f(x), α * (N² + ∂z_b))
            else
                ν = forcings.ν isa Function ? forcings.ν(x) : forcings.ν
            end
            abs_f = max(abs(params.f(x)), f_min)

            for i in 1:n_p, j in 1:n_p
                φᵢφⱼ   = shape_value(cv_p, q, i) * shape_value(cv_p, q, j)
                ∇φᵢ∇φⱼ = shape_gradient(cv_p, q, i) ⋅ shape_gradient(cv_p, q, j)
                M_cell[i, j]  += φᵢφⱼ * dΩ
                Mν_cell[i, j] += φᵢφⱼ / (2α²ε² * ν) * dΩ
                Kf_cell[i, j] += ∇φᵢ∇φⱼ / abs_f * dΩ
            end
        end

        dofs = celldofs(cell_up)[p_range]
        for (j, col) in enumerate(dofs), (i, row) in enumerate(dofs)
            push!(rows, row); push!(cols, col)
            push!(M_vals, M_cell[i, j])
            push!(Mν_vals, Mν_cell[i, j])
            push!(Kf_vals, Kf_cell[i, j])
        end
    end

    N = ndofs(dh_up)
    C = fe_data.C_up[:, fe_data.free_p]
    condense(vals) = C' * sparse(rows, cols, vals, N, N) * C
    return condense(M_vals), condense(Mν_vals), condense(Kf_vals)
end

"""
    P = BlockTriangularPreconditioner(arch, fe_data, params, forcings, A, b)

Block-triangular preconditioner for the condensed inversion matrix `A` at buoyancy
`b`, applied as `mul!(y, P, x)`. The velocity block is factorized by ILU(0) on the
GPU and by a sparse LU on the CPU; the Schur complement is approximated by
[`CahouetChabardInverse`](@ref).
"""
struct BlockTriangularPreconditioner{F, S, M, V}
    A_uu::F                  # factorization of the velocity block
    schur_inverse::S
    A_up::M
    free_u::UnitRange{Int}
    free_p::UnitRange{Int}
    u_rhs::V
end

function BlockTriangularPreconditioner(arch::AbstractArchitecture, fe_data::FEData,
                                       params::Parameters, forcings::Forcings,
                                       A::SparseMatrixCSC, b::AbstractVector)
    free_u, free_p = fe_data.free_u, fe_data.free_p
    A_uu = arch isa GPU ? KrylovPreconditioners.kp_ilu0(on_architecture(arch, A[free_u, free_u])) :
                          lu(A[free_u, free_u])
    return BlockTriangularPreconditioner(A_uu,
                                         CahouetChabardInverse(arch, fe_data, params, forcings, b),
                                         on_architecture(arch, A[free_u, free_p]),
                                         free_u, free_p,
                                         on_architecture(arch, zeros(length(free_u))))
end

function LinearAlgebra.mul!(y, P::BlockTriangularPreconditioner, x)
    x_u, x_p = view(x, P.free_u), view(x, P.free_p)
    y_u, y_p = view(y, P.free_u), view(y, P.free_p)
    mul!(y_p, P.schur_inverse, x_p)
    P.u_rhs .= x_u
    mul!(P.u_rhs, P.A_up, y_p, -1.0, 1.0)
    ldiv!(y_u, P.A_uu, P.u_rhs)
    return y
end
