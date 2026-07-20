"""
    EvolutionLHSCache

Pre-condensed evolution operators. Condensation is linear and pattern-
preserving, so the BC-condensed LHS for any θ is the allocation-free
combination `A.nzval = M_c.nzval + θ (Kₕ_c.nzval + Kᵥ_c.nzval)` (and likewise
`f_bc`), instead of a `copy(M)` + full `apply!` per timestep. `Kᵥ_c` must be
refreshed via `_refresh_Kᵥ!` whenever `Kᵥ` changes. Ferrite's `apply!` is used
here (not `condense_system`) because these operators are symmetric, where it
is exact and condenses in place.
"""
struct EvolutionLHSCache
    M_c::SparseMatrixCSC{Float64, Int}
    Kₕ_c::SparseMatrixCSC{Float64, Int}
    Kᵥ_c::SparseMatrixCSC{Float64, Int}
    f_bc_M::Vector{Float64}
    f_bc_Kₕ::Vector{Float64}
    f_bc_Kᵥ::Vector{Float64}
    A::SparseMatrixCSC{Float64, Int}   # CPU combination buffer (same pattern)
end

function EvolutionLHSCache(M, Kₕ, Kᵥ, ch_b)
    condense(K) = begin
        K_c = copy(K)
        f_bc = zeros(size(K, 1))
        apply!(K_c, f_bc, ch_b)   # symmetric matrices: Ferrite's apply! is exact
        K_c, f_bc
    end
    M_c,  f_bc_M  = condense(M)
    Kₕ_c, f_bc_Kₕ = condense(Kₕ)
    Kᵥ_c, f_bc_Kᵥ = condense(Kᵥ)
    return EvolutionLHSCache(M_c, Kₕ_c, Kᵥ_c, f_bc_M, f_bc_Kₕ, f_bc_Kᵥ, copy(M_c))
end

function _refresh_Kᵥ!(lhs::EvolutionLHSCache, Kᵥ::SparseMatrixCSC, ch_b)
    lhs.Kᵥ_c.nzval .= Kᵥ.nzval
    fill!(lhs.f_bc_Kᵥ, 0.0)
    apply!(lhs.Kᵥ_c, lhs.f_bc_Kᵥ, ch_b)
    return lhs
end

struct EvolutionToolkit{A<:AbstractArchitecture, M, V, S<:IterativeSolverToolkit}
    arch::A
    Kᵥ::M        # vertical stiffness (raw; rebuilt each step when conv_param is on)
    Kᵥ⁰::M       # static κᵥ part of Kᵥ (Kᵥ = Kᵥ⁰ + convection augmentation)
    lhs::EvolutionLHSCache
    rhs_diff::V  # -N² κᵥ ∂z(b) source term
    rhs_diff⁰::Vector{Float64}   # static κᵥ part of rhs_diff (CPU)
    rhs_flux::V  # α F surface flux
    f_bc::V      # RHS correction for inhomogeneous Dirichlet BCs (0 when BCs are homogeneous)
    solver::S
end

function Base.summary(evolution::EvolutionToolkit)
    t = typeof(evolution)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, evolution::EvolutionToolkit)
    println(io, summary(evolution), ":")
    println(io, "├── arch: ", evolution.arch)
    println(io, "├── Kᵥ: ", summary(evolution.Kᵥ))
    println(io, "├── rhs_diff: ", summary(evolution.rhs_diff))
    println(io, "├── rhs_flux: ", summary(evolution.rhs_flux))
    println(io, "├── f_bc: ", summary(evolution.f_bc))
      print(io, "└── solver: ", summary(evolution.solver))
end

"""
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings, ts; kwargs...)

Set up the toolkit for the buoyancy evolution equation

    μϱ (∂ₜb + u·∇b) = α²ε² [∇ₕ·(κₕ∇ₕb) + ∂z(κᵥ ∂z b)]
"""
function EvolutionToolkit(arch::AbstractArchitecture,
                          fe_data::FEData,
                          params::Parameters,
                          forcings::Forcings,
                          ts::AbstractTimestepper;
                          atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=false)
    κₕ = forcings.κₕ
    κᵥ = forcings.κᵥ

    @info "Building evolution system..."

    M        = build_M(fe_data)
    Kₕ       = build_Kₕ(fe_data, κₕ)
    Kᵥ⁰      = build_Kᵥ(fe_data, κᵥ)
    Kᵥ       = copy(Kᵥ⁰)
    lhs      = EvolutionLHSCache(M, Kₕ, Kᵥ, fe_data.ch_b)
    rhs_diff⁰ = build_rhs_diff(params, fe_data, κᵥ)
    rhs_diff = copy(rhs_diff⁰)
    rhs_flux = build_rhs_flux(params, forcings, fe_data)

    # θ from a BDF1 startup step, but the preconditioner type must match the real
    # timestepper: adaptive runs later rebuild with a Diagonal preconditioner, and
    # the solver's parametric type is fixed by this first assignment.
    ts1 = BDF1(; ts.t_start, t_stop=ts.t_stop, Δt=ts.Δt[])
    A, P, f_bc = _combine_evolution_LHS(arch, params, forcings, ts1, lhs;
                                        use_diag=_use_diag_precond(arch, forcings, ts))

    rhs_diff = on_architecture(arch, rhs_diff)
    rhs_flux = on_architecture(arch, rhs_flux)
    f_bc     = on_architecture(arch, f_bc)

    N  = size(A, 1)
    T  = eltype(A)
    y  = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    workspace = Krylov.CgWorkspace(N, N, VT)
    workspace.x .= zero(T)

    verbose_int = verbose ? 1 : 0
    kwargs_dict = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)
    solver = IterativeSolverToolkit(A, P, y, workspace, kwargs_dict, "Evolution")

    return EvolutionToolkit(arch, Kᵥ, Kᵥ⁰, lhs, rhs_diff, rhs_diff⁰, rhs_flux, f_bc, solver)
end

"""
    collect_evolution_LHS!(evolution, params, forcings, ts, ch_b; Kᵥ_changed=true)

Re-form the evolution LHS `A = M + θ(Kₕ + Kᵥ)` (BC-condensed) for the current
`Δt` by combining the cached condensed operators. Pass `Kᵥ_changed=false` when
`Kᵥ` is unchanged (e.g. adaptive-Δt rebuilds without the convection
parameterization) to skip re-condensing it.
"""
function collect_evolution_LHS!(evolution::EvolutionToolkit, params::Parameters,
                                 forcings::Forcings, ts::AbstractTimestepper, ch_b;
                                 Kᵥ_changed::Bool = true)
    arch = evolution.arch
    Kᵥ_changed && _refresh_Kᵥ!(evolution.lhs, evolution.Kᵥ, ch_b)
    A, P, f_bc = _combine_evolution_LHS(arch, params, forcings, ts, evolution.lhs)
    evolution.solver.A = A
    evolution.solver.P = P
    evolution.f_bc    .= on_architecture(arch, f_bc)
    return evolution
end

"""
    update_evolution_LHS!(evolution, fe_data, params, forcings, ts, b_cpu)

Per-step operator refresh: when the convection parameterization is on, rebuild
`Kᵥ` and `rhs_diff` from the current buoyancy, then re-form the condensed LHS
if anything invalidated it (`Kᵥ` changed or `Δt` is adaptive).
"""
function update_evolution_LHS!(evolution::EvolutionToolkit, fe_data::FEData,
                               params::Parameters, forcings::Forcings,
                               ts::AbstractTimestepper, b_cpu::AbstractVector)
    conv_on = forcings.conv_param.is_on
    if conv_on
        @ctime "  build Kᵥ" build_Kᵥ_conv!(evolution.Kᵥ, evolution.Kᵥ⁰, fe_data, params,
                                           forcings.conv_param, b_cpu)
        @ctime "  build rhs_diff" begin
            rhs_diff_new = build_rhs_diff_conv!(zeros(fe_data.nb), evolution.rhs_diff⁰,
                                                fe_data, params, forcings.conv_param, b_cpu)
            evolution.rhs_diff .= on_architecture(evolution.arch, rhs_diff_new)
        end
    end
    if conv_on || ts.adaptive
        collect_evolution_LHS!(evolution, params, forcings, ts, fe_data.ch_b;
                               Kᵥ_changed=conv_on)
    end
    return evolution
end

function _combine_evolution_LHS(arch::AbstractArchitecture, params::Parameters,
                                 forcings::Forcings, ts::AbstractTimestepper,
                                 lhs::EvolutionLHSCache;
                                 use_diag = _use_diag_precond(arch, forcings, ts))
    θ = evolution_parameter(params, ts)
    A = lhs.A
    @. A.nzval = lhs.M_c.nzval + θ * (lhs.Kₕ_c.nzval + lhs.Kᵥ_c.nzval)
    f_bc = @. lhs.f_bc_M + θ * (lhs.f_bc_Kₕ + lhs.f_bc_Kᵥ)

    if use_diag
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $(size(A, 1)) DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end

    return on_architecture(arch, A), P, f_bc
end

# adaptive Δt, per-step Kᵥ rebuilds, and GPU runs all preclude a one-time LU
# (BDF2.adaptive is currently always false)
_use_diag_precond(arch, forcings, ts) =
    typeof(arch) == GPU || forcings.conv_param.is_on || ts.adaptive

"""
    θ = evolution_parameter(params, ts)

Coefficient θ in `A = M + θ*(Kₕ + Kᵥ)`.
"""
function evolution_parameter(p::Parameters, ts::BDF1)
    return ts.Δt[] * p.α^2 * p.ε^2 / p.μϱ
end
function evolution_parameter(p::Parameters, ts::BDF2)
    return 2/3 * ts.Δt[] * p.α^2 * p.ε^2 / p.μϱ
end

####
#### Matrix and vector builders
####

"""
    M = build_M(fe_data)

Assemble the buoyancy mass matrix `M = ∫ φᵢ φⱼ dΩ`.
"""
function build_M(fe_data::FEData)
    dh_b  = fe_data.dh_b
    _, _, cv_b = make_cell_values(fe_data)
    n_b   = getnbasefunctions(cv_b)

    M = allocate_evolution_matrix(fe_data)
    asm = start_assemble(M)
    Me  = zeros(n_b, n_b)

    for cc in CellIterator(dh_b)
        reinit!(cv_b, cc)
        fill!(Me, 0.0)
        for q in 1:getnquadpoints(cv_b)
            dΩ = getdetJdV(cv_b, q)
            for i in 1:n_b
                φᵢ = shape_value(cv_b, q, i)
                for j in 1:n_b
                    Me[i, j] += φᵢ * shape_value(cv_b, q, j) * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cc), Me)
    end
    return M
end

"""
    Kₕ = build_Kₕ(fe_data, κₕ)

Assemble the horizontal stiffness `Kₕ = ∫ κₕ (∂x φᵢ ∂x φⱼ + ∂y φᵢ ∂y φⱼ) dΩ`.
"""
function build_Kₕ(fe_data::FEData, κₕ)
    dh_b  = fe_data.dh_b
    _, _, cv_b = make_cell_values(fe_data)
    n_b   = getnbasefunctions(cv_b)

    Kₕ = allocate_evolution_matrix(fe_data)
    asm = start_assemble(Kₕ)
    Ke  = zeros(n_b, n_b)

    for cc in CellIterator(dh_b)
        reinit!(cv_b, cc)
        coords = getcoordinates(cc)
        fill!(Ke, 0.0)
        for q in 1:getnquadpoints(cv_b)
            x = spatial_coordinate(cv_b, q, coords)
            κ = κₕ(x)
            dΩ = getdetJdV(cv_b, q)
            for i in 1:n_b
                ∇φᵢ = shape_gradient(cv_b, q, i)
                for j in 1:n_b
                    ∇φⱼ = shape_gradient(cv_b, q, j)
                    Ke[i, j] += κ * (∇φᵢ[1]*∇φⱼ[1] + ∇φᵢ[2]*∇φⱼ[2]) * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cc), Ke)
    end
    return Kₕ
end

"""
    Kᵥ = build_Kᵥ(fe_data, κᵥ)

Assemble the vertical stiffness `Kᵥ = ∫ κᵥ ∂z φᵢ ∂z φⱼ dΩ`.
"""
function build_Kᵥ(fe_data::FEData, κᵥ)
    dh_b  = fe_data.dh_b
    _, _, cv_b = make_cell_values(fe_data)
    n_b   = getnbasefunctions(cv_b)

    Kᵥ = allocate_evolution_matrix(fe_data)
    asm = start_assemble(Kᵥ)
    Ke  = zeros(n_b, n_b)

    for cc in CellIterator(dh_b)
        reinit!(cv_b, cc)
        coords = getcoordinates(cc)
        fill!(Ke, 0.0)
        for q in 1:getnquadpoints(cv_b)
            x = spatial_coordinate(cv_b, q, coords)
            κ = κᵥ(x)
            dΩ = getdetJdV(cv_b, q)
            for i in 1:n_b
                ∂zφᵢ = shape_gradient(cv_b, q, i)[3]
                for j in 1:n_b
                    Ke[i, j] += κ * ∂zφᵢ * shape_gradient(cv_b, q, j)[3] * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cc), Ke)
    end
    return Kᵥ
end

"""
    f = build_rhs_diff(params, fe_data, κᵥ)

Assemble the background-stratification source `f = ∫ -N² κᵥ ∂z φᵢ dΩ`.
"""
function build_rhs_diff(params::Parameters, fe_data::FEData, κᵥ)
    N²   = params.N²
    dh_b = fe_data.dh_b
    _, _, cv_b = make_cell_values(fe_data)
    n_b  = getnbasefunctions(cv_b)

    f  = zeros(fe_data.nb)
    fₑ = zeros(n_b)

    for cc in CellIterator(dh_b)
        reinit!(cv_b, cc)
        coords = getcoordinates(cc)
        fill!(fₑ, 0.0)
        for q in 1:getnquadpoints(cv_b)
            x = spatial_coordinate(cv_b, q, coords)
            κ = κᵥ(x)
            dΩ = getdetJdV(cv_b, q)
            for i in 1:n_b
                fₑ[i] -= N² * κ * shape_gradient(cv_b, q, i)[3] * dΩ
            end
        end
        f[celldofs(cc)] .+= fₑ
    end
    return f
end

"""
    f = build_rhs_flux(params, forcings, fe_data)

Dispatch on surface boundary condition type.
"""
build_rhs_flux(params::Parameters, forcings::Forcings, fe_data::FEData) =
    build_rhs_flux(params, fe_data, forcings.b_surface_bc)

function build_rhs_flux(params::Parameters, fe_data::FEData, bc::SurfaceFluxBC)
    α    = params.α
    dh_b = fe_data.dh_b
    _, fv_b = make_facet_values(fe_data)
    n_b  = getnbasefunctions(fv_b)
    facetset = fe_data.mesh.grid.facetsets[fe_data.mesh.surface_tag]

    f  = zeros(fe_data.nb)
    fₑ = zeros(n_b)

    for fc in FacetIterator(dh_b, facetset)
        reinit!(fv_b, fc)
        coords = getcoordinates(fc)
        fill!(fₑ, 0.0)
        for q in 1:getnquadpoints(fv_b)
            x = spatial_coordinate(fv_b, q, coords)
            dΓ = getdetJdV(fv_b, q)
            for i in 1:n_b
                fₑ[i] += α * bc.flux(x) * shape_value(fv_b, q, i) * dΓ
            end
        end
        f[celldofs(fc)] .+= fₑ
    end
    return f
end

function build_rhs_flux(params::Parameters, fe_data::FEData, bc::SurfaceDirichletBC)
    return zeros(fe_data.nb)
end

####
#### Advection RHS (assembled from current DOF vectors each timestep)
####

"""
    f = build_rhs_adv(fe_data, params, u_vec, b_vec, ts::BDF1)
    f = build_rhs_adv(fe_data, params, u_vec, b_vec, u_prev, b_prev, ts::BDF2)

Assemble the advection right-hand side.

BDF1: `∫ (b - Δt (u·∇b + w N²)) d dΩ`
BDF2: `∫ (4/3 b - 1/3 b_prev - 2/3 Δt ((2u-u_prev)·∇(2b-b_prev) + (2w-w_prev) N²)) d dΩ`

Runs on the `AssemblyCache` (reference shape tables + per-cell geometry/DOF maps),
so the per-step cost is a tight allocation-free loop with no `reinit!`.
"""
build_rhs_adv(fe_data, params, u_vec, b_vec, u_prev, b_prev, ts::BDF1) =
    build_rhs_adv(fe_data, params, u_vec, b_vec, ts)

function build_rhs_adv(fe_data::FEData, params::Parameters,
                        u_vec::AbstractVector, b_vec::AbstractVector,
                        ts::BDF1)
    f = zeros(fe_data.nb)
    _rhs_adv_kernel!(f, fe_data.cache, params.N²,
                     (b, bp) -> b, (b, bp) -> b, (u, up) -> u, 1.0, ts.Δt[],
                     u_vec, b_vec, u_vec, b_vec)
    return f
end

function build_rhs_adv(fe_data::FEData, params::Parameters,
                        u_vec::AbstractVector, b_vec::AbstractVector,
                        u_prev::AbstractVector, b_prev::AbstractVector,
                        ts::BDF2)
    f = zeros(fe_data.nb)
    _rhs_adv_kernel!(f, fe_data.cache, params.N²,
                     (b, bp) -> 4/3*b - 1/3*bp, (b, bp) -> 2*b - bp, (u, up) -> 2*u - up,
                     2/3, ts.Δt[], u_vec, b_vec, u_prev, b_prev)
    return f
end

"""
    _rhs_adv_kernel!(f, cache, N², b_comb, b_eff, u_eff, Δt_fac, Δt,
                     u_vec, b_vec, u_prev, b_prev)

Shared advection-RHS kernel: accumulates
`∫ (b_comb(b, b_prev) - Δt_fac Δt (u_eff·∇(b_eff) + w_eff N²)) φᵢ dΩ`
with the pointwise combinations `b_comb`, `b_eff`, `u_eff` supplied per scheme.
"""
function _rhs_adv_kernel!(f::Vector{Float64}, cache::AssemblyCache, N²,
                          b_comb::F1, b_eff::F2, u_eff::F3, Δt_fac, Δt,
                          u_vec::AbstractVector, b_vec::AbstractVector,
                          u_prev::AbstractVector, b_prev::AbstractVector) where {F1, F2, F3}
    nq   = length(cache.w)
    n_b  = size(cache.phi_b, 2)
    n_su = size(cache.phi_u, 2)
    ncells = length(cache.detJ)

    lb  = zeros(n_b)        # b
    lbp = zeros(n_b)        # b_prev
    U   = zeros(3, n_su)    # effective velocity at scalar-basis nodes
    fₑ  = zeros(n_b)

    @inbounds for c in 1:ncells
        for i in 1:n_b
            d = cache.dofs_b[i, c]
            lb[i]  = b_vec[d]
            lbp[i] = b_prev[d]
        end
        for j in 1:n_su, k in 1:3
            d = cache.dofs_u[3*(j - 1) + k, c]
            U[k, j] = u_eff(u_vec[d], u_prev[d])
        end
        Jᵀ = cache.Jinv_t[c]
        dJ = cache.detJ[c]
        fill!(fₑ, 0.0)
        for q in 1:nq
            # b, b_prev, and reference gradient of b_eff
            b_q = 0.0; bp_q = 0.0
            g1 = 0.0; g2 = 0.0; g3 = 0.0
            for i in 1:n_b
                b_q  += cache.phi_b[q, i] * lb[i]
                bp_q += cache.phi_b[q, i] * lbp[i]
                beff  = b_eff(lb[i], lbp[i])
                g1 += cache.dphi_b[1, i, q] * beff
                g2 += cache.dphi_b[2, i, q] * beff
                g3 += cache.dphi_b[3, i, q] * beff
            end
            ∇b1 = _∂ᵣ(Jᵀ, 1, g1, g2, g3)
            ∇b2 = _∂ᵣ(Jᵀ, 2, g1, g2, g3)
            ∇b3 = _∂ᵣ(Jᵀ, 3, g1, g2, g3)
            # effective velocity
            u1 = 0.0; u2 = 0.0; u3 = 0.0
            for j in 1:n_su
                φ = cache.phi_u[q, j]
                u1 += φ * U[1, j]; u2 += φ * U[2, j]; u3 += φ * U[3, j]
            end
            adv   = u1*∇b1 + u2*∇b2 + u3*∇b3 + u3*N²
            rhs_q = (b_comb(b_q, bp_q) - Δt_fac * Δt * adv) * cache.w[q] * dJ
            for i in 1:n_b
                fₑ[i] += rhs_q * cache.phi_b[q, i]
            end
        end
        for i in 1:n_b
            f[cache.dofs_b[i, c]] += fₑ[i]
        end
    end
    return f
end

####
#### Parametrization-aware rebuilds (conv_param: κᵥ depends on ∂z(b))
####

"""
    Kᵥ = build_Kᵥ_conv(fe_data, params, forcings, b_vec)
    build_Kᵥ_conv!(Kᵥ, Kᵥ⁰, fe_data, params, conv_param, b_vec)

Rebuild `Kᵥ` with the convection parameterization: `Kᵥ = Kᵥ⁰ + Kᶜ(b)`, where
`Kᵥ⁰` is the static background-`κᵥ` stiffness and `Kᶜ` uses the augmentation
`_κ_conv_extra` evaluated from `∂z(b)` at each quadrature point. The split
means the (possibly space-dependent) background `κᵥ` is never re-evaluated.
The in-place version scatters directly into `Kᵥ.nzval` via the cached sparsity
index map; `Kᵥ` and `Kᵥ⁰` must both have the evolution pattern.
"""
function build_Kᵥ_conv(fe_data::FEData, params::Parameters,
                       forcings::Forcings, b_vec::AbstractVector)
    Kᵥ⁰ = build_Kᵥ(fe_data, forcings.κᵥ)
    return build_Kᵥ_conv!(copy(Kᵥ⁰), Kᵥ⁰, fe_data, params, forcings.conv_param, b_vec)
end

function build_Kᵥ_conv!(Kᵥ::SparseMatrixCSC, Kᵥ⁰::SparseMatrixCSC, fe_data::FEData,
                        params::Parameters, conv_param::ConvectionParameterization,
                        b_vec::AbstractVector)
    cache = fe_data.cache
    nq   = length(cache.w)
    n_b  = size(cache.phi_b, 2)
    ncells = length(cache.detJ)
    α  = params.α
    N² = params.N²

    Kᵥ.nzval .= Kᵥ⁰.nzval
    lb  = zeros(n_b)
    ∂zφ = zeros(n_b)
    Ke  = zeros(n_b, n_b)

    @inbounds for c in 1:ncells
        for i in 1:n_b
            lb[i] = b_vec[cache.dofs_b[i, c]]
        end
        Jᵀ = cache.Jinv_t[c]
        dJ = cache.detJ[c]
        fill!(Ke, 0.0)
        for q in 1:nq
            g1 = 0.0; g2 = 0.0; g3 = 0.0
            for i in 1:n_b
                g1 += cache.dphi_b[1, i, q] * lb[i]
                g2 += cache.dphi_b[2, i, q] * lb[i]
                g3 += cache.dphi_b[3, i, q] * lb[i]
                ∂zφ[i] = _∂ᵣ(Jᵀ, 3, cache.dphi_b[1, i, q], cache.dphi_b[2, i, q],
                             cache.dphi_b[3, i, q])
            end
            ∂z_b_q = _∂ᵣ(Jᵀ, 3, g1, g2, g3)
            κdΩ = _κ_conv_extra(conv_param, α * (N² + ∂z_b_q)) * cache.w[q] * dJ
            for j in 1:n_b, i in 1:n_b
                Ke[i, j] += κdΩ * ∂zφ[i] * ∂zφ[j]
            end
        end
        for j in 1:n_b, i in 1:n_b
            Kᵥ.nzval[cache.nzidx_b[n_b*(j - 1) + i, c]] += Ke[i, j]
        end
    end
    return Kᵥ
end

"""
    f = build_rhs_diff_conv(params, fe_data, forcings, b_vec)
    f = build_rhs_diff_conv!(f, rhs_diff⁰, fe_data, params, conv_param, b_vec)

Rebuild `rhs_diff` with the convection parameterization, as the static
background part plus the `_κ_conv_extra` augmentation (cached kernel).
"""
function build_rhs_diff_conv(params::Parameters, fe_data::FEData,
                              forcings::Forcings, b_vec::AbstractVector)
    rhs_diff⁰ = build_rhs_diff(params, fe_data, forcings.κᵥ)
    return build_rhs_diff_conv!(zeros(fe_data.nb), rhs_diff⁰, fe_data, params,
                                forcings.conv_param, b_vec)
end

function build_rhs_diff_conv!(f::Vector{Float64}, rhs_diff⁰::Vector{Float64},
                              fe_data::FEData, params::Parameters,
                              conv_param::ConvectionParameterization,
                              b_vec::AbstractVector)
    cache = fe_data.cache
    nq   = length(cache.w)
    n_b  = size(cache.phi_b, 2)
    ncells = length(cache.detJ)
    α  = params.α
    N² = params.N²

    f .= rhs_diff⁰
    lb = zeros(n_b)
    fₑ = zeros(n_b)

    @inbounds for c in 1:ncells
        for i in 1:n_b
            lb[i] = b_vec[cache.dofs_b[i, c]]
        end
        Jᵀ = cache.Jinv_t[c]
        dJ = cache.detJ[c]
        fill!(fₑ, 0.0)
        for q in 1:nq
            g1 = 0.0; g2 = 0.0; g3 = 0.0
            for i in 1:n_b
                g1 += cache.dphi_b[1, i, q] * lb[i]
                g2 += cache.dphi_b[2, i, q] * lb[i]
                g3 += cache.dphi_b[3, i, q] * lb[i]
            end
            ∂z_b_q = _∂ᵣ(Jᵀ, 3, g1, g2, g3)
            κdΩ = _κ_conv_extra(conv_param, α * (N² + ∂z_b_q)) * cache.w[q] * dJ
            for i in 1:n_b
                ∂zφᵢ = _∂ᵣ(Jᵀ, 3, cache.dphi_b[1, i, q], cache.dphi_b[2, i, q],
                           cache.dphi_b[3, i, q])
                fₑ[i] -= N² * κdΩ * ∂zφᵢ
            end
        end
        for i in 1:n_b
            f[cache.dofs_b[i, c]] += fₑ[i]
        end
    end
    return f
end
