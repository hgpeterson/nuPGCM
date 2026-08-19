####
#### Pressure-space operators for Schur-complement preconditioning
####
####    Everything in this file lives on the *reduced* pressure space (the free
####    pressure DOFs of the condensed inversion system), so the matrices here
####    can be used directly as `S̃` approximations for the (2,2) block.
####
#### Why these particular operators exist -- the rotating-Stokes Schur symbol.
####
#### The inversion LHS is
####
####     𝒜 = [ K + N   -Bᵀ ]      K = ∫ 2η ε(u):ε(v),  η = α²ε²ν   (SPD)
####         [ B        0  ]      N = ∫ f (ẑ×u)·v                  (skew-symmetric)
####                              B = ∫ q (∇·u)
####
#### and the pressure Schur complement is S = B A⁻¹ Bᵀ with A = K + N. Freezing
#### the coefficients and Fourier-transforming with wavevector k, A has symbol
#### η(|k|²I + kkᵀ) + f J with J = ẑ× (skew). Sherman--Morrison on the rank-1
#### part gives S(k) = s₀/(1 + η s₀), where, writing a = η|k|² and k_h² = k₁²+k₂²,
####
####                a k_h²      k₃²
####     s₀(k)  =  ───────  +  ───
####                a² + f²      a
####
#### The skew part contributes *nothing* at leading order: its two cross terms are
#### k₁(f k₂) + k₂(-f k₁) = 0. That exact cancellation is geostrophic degeneracy,
#### and it is what breaks the standard preconditioner:
####
####     viscous limit   a ≫ f :  S → 1/(2η)                        ⇒ mass matrix ✓
####                                1   k₃²       η
####     rotating limit  a ≪ f :  S → ─── ──── + ─── k_h²|k|²       ⇒ mass matrix ✗
####                                η   |k|²      f²
####
#### i.e. the pressure mass matrix overestimates S by |k|²/k₃², which blows up for
#### depth-independent (barotropic) pressure modes. Physically: a barotropic p
#### drives a geostrophic flow that is already divergence-free, so it never feels
#### the incompressibility constraint; only β and Ekman pumping restore it.
####
#### The operators assembled below are exactly the pieces needed to build S̃ with
#### that k₃²/|k|² factor restored (see `GeostrophicSchur` in preconditioners.jl):
####
####     S̃ = 𝒦 K_p⁻¹ M_p ,   𝒦 = Kz_ην + Kh_ν_f2 M_p⁻¹K_p M_p⁻¹K_p
####
#### whose symbol is [k₃²/η + (η k_h²/f²)|k|⁴] / |k|², matching s₀ above.
####
#### A caveat worth stating in advance, because it bounds what any of this can
#### achieve. The frozen-coefficient symbol assumes constant f. With f varying
#### (here f = y, so β = 1), the barotropic block of S picks up the divergence of
#### the geostrophic flow, ∇·((1/f)ẑ×∇p) = -(β/f²)∂ₓp. But its quadratic form
####
####     ⟨p, -(β/f²)∂ₓp⟩ = -∫ (β/f²) ∂ₓ(p²/2) = 0
####
#### vanishes identically, because β/f² has no x-dependence. So the barotropic
#### block of S is, to leading order in β, *purely skew-symmetric and O(β)*: its
#### symmetric part comes only from viscosity and Ekman pumping. Eigenvalues of S
#### sitting near the imaginary axis with small modulus are intrinsically hard for
#### GMRES and cannot be fixed by *any* symmetric positive-definite S̃.
####
#### The practical consequence: `δ` in the anisotropic-Poisson variant is a
#### stand-in for whatever lifts the barotropic modes off zero (β and Ekman
#### friction), and its best value has to be found empirically. Expect the
#### geostrophic approximation to buy a large factor over the mass matrix, but not
#### mesh-independent single-digit iteration counts.
####

"""
    PressureOperators

Assembled pressure-space matrices used to build Schur-complement approximations,
all restricted to the free pressure DOFs of the condensed system (`np_free ×
np_free`, CPU `SparseMatrixCSC`). Symbols are quoted for frozen coefficients.

| field     | form                                   | symbol           |
|-----------|----------------------------------------|------------------|
| `M`       | `∫ p q`                                | `1`              |
| `Mν`      | `∫ p q / (2η)`                         | `1/(2η)`         |
| `K`       | `∫ ∇p·∇q`                              | `\\|k\\|²`         |
| `Kz_ην`   | `∫ ∂z p ∂z q / η`                      | `k₃²/η`          |
| `Kh_ην`   | `∫ ∇ₕp·∇ₕq / η`                        | `k_h²/η`         |
| `Kh_ν_f2` | `∫ (η/f²) ∇ₕp·∇ₕq`                     | `η k_h²/f²`      |
| `Kf`      | `∫ (1/\\|f\\|) ∇p·∇q`                    | `\\|k\\|²/\\|f\\|`   |

with `η = α²ε²ν`. `f` is floored at `f_min` in magnitude, because `f(x) = x[2]`
vanishes at the equator in the channel-basin setup and the `1/f` weights would
otherwise be singular there.

`Mlump` is the lumped (row-summed) diagonal of `M`, used wherever an `M⁻¹` shows
up inside a composed operator — exact inversion of `M` there would cost more than
it is worth, and lumping is spectrally equivalent for P1.
"""
struct PressureOperators
    M::SparseMatrixCSC{Float64, Int}
    Mν::SparseMatrixCSC{Float64, Int}
    K::SparseMatrixCSC{Float64, Int}
    Kz_ην::SparseMatrixCSC{Float64, Int}
    Kh_ην::SparseMatrixCSC{Float64, Int}
    Kh_ν_f2::SparseMatrixCSC{Float64, Int}
    Kf::SparseMatrixCSC{Float64, Int}
    Mlump::Vector{Float64}
    f_min::Float64
end

function Base.summary(po::PressureOperators)
    t = typeof(po)
    return "$(parentmodule(t)).$(nameof(t))($(size(po.M, 1)) pressure DOFs)"
end

"""
    C_p = pressure_constraint_matrix(fe_data)

The pressure sub-block of `fe_data.C_up`: an `np × np_free` map taking reduced
free pressure DOFs to full-length pressure DOFs, so that a full-space pressure
operator `X` condenses as `C_p' * X * C_p`.

Velocity and pressure constraints never mix (Dirichlet and periodic constraints
are declared per field), so the pressure columns of `C_up` have nonzeros only in
pressure rows and this slice loses nothing.
"""
function pressure_constraint_matrix(fe_data::FEData)
    return fe_data.C_up[fe_data.p_dof_indices, fe_data.p_free_red]
end

"""
    po = build_pressure_operators(fe_data, params, forcings; b_vec, f_min)

Assemble every matrix in [`PressureOperators`](@ref) in a single sweep over the
mesh, then condense onto the free pressure DOFs.

The viscosity `η = α²ε²ν` is evaluated at each quadrature point. When the eddy
parameterization is active, `ν` comes from `ν_eddy` at `α(N² + ∂z b)` using
`b_vec` (defaulting to zeros), exactly as in [`build_A_visc!`](@ref), so the
preconditioner sees the same `ν` field as the operator it preconditions.
"""
function build_pressure_operators(fe_data::FEData, params::Parameters, forcings::Forcings;
                                  b_vec::Union{Nothing, AbstractVector} = nothing,
                                  f_min::Real = 1e-2)
    dh_up  = fe_data.dh_up
    dh_b   = fe_data.dh_b
    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    n_p    = getnbasefunctions(cv_p)
    α²ε²   = params.α^2 * params.ε^2
    α, N²  = params.α, params.N²
    f_cor  = params.f
    eddy   = forcings.eddy_param
    np     = fe_data.np

    b_vec === nothing && (b_vec = zeros(fe_data.nb))

    # global pressure DOF -> position within p_dof_indices
    p_pos = zeros(Int, ndofs(dh_up))
    for (j, i) in enumerate(fe_data.p_dof_indices)
        p_pos[i] = j
    end
    range_p = dof_range(dh_up, :p)

    # one COO triplet set per operator
    nops   = 7
    Is     = [Int[]     for _ in 1:nops]
    Js     = [Int[]     for _ in 1:nops]
    Vs     = [Float64[] for _ in 1:nops]
    Ae     = [zeros(n_p, n_p) for _ in 1:nops]

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_u, cc_up)
        reinit!(cv_p, cc_up)
        reinit!(cv_b, cc_b)
        coords  = getcoordinates(cc_up)
        local_b = b_vec[celldofs(cc_b)]
        for e in Ae; fill!(e, 0.0); end

        for q in 1:getnquadpoints(cv_p)
            x  = spatial_coordinate(cv_p, q, coords)
            dΩ = getdetJdV(cv_p, q)

            if eddy.is_on
                ∂z_b = function_gradient(cv_b, q, local_b)[3]
                ν_q  = ν_eddy(eddy, eddy.f(x), α * (N² + ∂z_b))
            else
                ν_q = forcings.ν isa Function ? forcings.ν(x) : forcings.ν
            end
            η_q  = α²ε² * ν_q
            f_q  = f_cor(x)
            fa   = max(abs(f_q), f_min)

            w = (1.0,               # M
                 1 / (2η_q),        # Mν
                 1.0,               # K       (full gradient)
                 1 / η_q,           # Kz_ην   (vertical only)
                 1 / η_q,           # Kh_ην   (horizontal only)
                 η_q / fa^2,        # Kh_ν_f2 (horizontal only)
                 1 / fa)            # Kf      (full gradient)

            for i in 1:n_p
                φi = shape_value(cv_p, q, i)
                ∇i = shape_gradient(cv_p, q, i)
                for j in 1:n_p
                    φj = shape_value(cv_p, q, j)
                    ∇j = shape_gradient(cv_p, q, j)
                    mass = φi * φj * dΩ
                    grad = (∇i ⋅ ∇j) * dΩ
                    gz   = ∇i[3] * ∇j[3] * dΩ
                    gh   = (∇i[1] * ∇j[1] + ∇i[2] * ∇j[2]) * dΩ
                    Ae[1][i, j] += w[1] * mass
                    Ae[2][i, j] += w[2] * mass
                    Ae[3][i, j] += w[3] * grad
                    Ae[4][i, j] += w[4] * gz
                    Ae[5][i, j] += w[5] * gh
                    Ae[6][i, j] += w[6] * gh
                    Ae[7][i, j] += w[7] * grad
                end
            end
        end

        dofs = celldofs(cc_up)[range_p]
        for (li, gi) in enumerate(dofs), (lj, gj) in enumerate(dofs)
            ri, rj = p_pos[gi], p_pos[gj]
            for o in 1:nops
                v = Ae[o][li, lj]
                iszero(v) && continue
                push!(Is[o], ri); push!(Js[o], rj); push!(Vs[o], v)
            end
        end
    end

    C_p  = pressure_constraint_matrix(fe_data)
    cond(o) = C_p' * sparse(Is[o], Js[o], Vs[o], np, np) * C_p

    M       = cond(1)
    Mν      = cond(2)
    K       = cond(3)
    Kz_ην   = cond(4)
    Kh_ην   = cond(5)
    Kh_ν_f2 = cond(6)
    Kf      = cond(7)

    return PressureOperators(M, Mν, K, Kz_ην, Kh_ην, Kh_ν_f2, Kf,
                             vec(sum(M, dims = 2)), Float64(f_min))
end

"""
    Mu_lump = build_velocity_mass_lumped(fe_data)

Lumped velocity mass matrix diagonal on the free velocity DOFs, i.e. the row sums
of `∫ φᵢ·φⱼ` condensed by the velocity block of `C_up`. This is the `T` weight in
the least-squares commutator (LSC/BFBt) Schur approximation, where only a cheap
spectrally-equivalent stand-in for `M_u⁻¹` is needed.
"""
function build_velocity_mass_lumped(fe_data::FEData)
    dh_up = fe_data.dh_up
    cv_u, _, _ = make_cell_values(fe_data)
    n_u   = getnbasefunctions(cv_u)
    N_up  = ndofs(dh_up)

    rows = Int[]; cols = Int[]; vals = Float64[]
    Me   = zeros(n_u, n_u)
    range_u = dof_range(dh_up, :u)

    for cc in CellIterator(dh_up)
        reinit!(cv_u, cc)
        fill!(Me, 0.0)
        for q in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q)
            for i in 1:n_u, j in 1:n_u
                Me[i, j] += (shape_value(cv_u, q, i) ⋅ shape_value(cv_u, q, j)) * dΩ
            end
        end
        dofs = celldofs(cc)[range_u]
        for (li, gi) in enumerate(dofs), (lj, gj) in enumerate(dofs)
            v = Me[li, lj]
            iszero(v) && continue
            push!(rows, gi); push!(cols, gj); push!(vals, v)
        end
    end

    Mu_full = sparse(rows, cols, vals, N_up, N_up)
    C_u     = fe_data.C_up[:, fe_data.u_free_red]
    Mu      = C_u' * Mu_full * C_u
    return vec(sum(Mu, dims = 2))
end
