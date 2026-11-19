"""
    EvolutionLHSCache(fe_data, C, A, CᵀMC, CᵀK⁰C, θ)

Buffers for forming the evolution matrix `A = Cᵀ(M + θ(K⁰ + Kᶜ))C` in place, on the
architecture of `A`, from `CᵀMC` and `CᵀK⁰C` on the CPU, where `K⁰ = Kₕ + Kᵥ⁰`, `Kᶜ` is
the convective part of `Kᵥ` and `C` is the constraint map. Each row of `C` has at most
one entry, equal to 1.

- `M`, `K⁰`: `nonzeros(CᵀMC)` and `nonzeros(CᵀK⁰C)` in the order of `nonzeros(A)`
- `diagonal`: position in `nonzeros(A)` of each diagonal entry
- `nzidx`: position in `nonzeros(A)` of entry `(i, j)` of the buoyancy block of cell
  `c` at `nzidx[n*(j-1)+i, c]`, with `n` buoyancy DOFs per cell, or 0 if the entry is
  eliminated
- `κdΩ`: `κᶜ w |det J|` at each quadrature point of each cell
- `θ`: the value `A` was last formed for
"""
struct EvolutionLHSCache{V, D, I, Q}
    M::V
    K⁰::V
    diagonal::D
    nzidx::I
    κdΩ::Q
    θ::Base.RefValue{Float64}
end

function EvolutionLHSCache(fe_data::FEData, C, A, CᵀMC, CᵀK⁰C, θ)
    arch     = architecture(A)
    cache    = fe_data.cache
    A_cpu    = on_architecture(CPU(), A)
    position = nonzero_positions(A)
    order    = invperm(position)
    free_b   = free_index(C)
    nzidx    = [k == 0 ? Int32(0) : Int32(position[k])
                for k in build_nzidx_map(A_cpu, free_b[cache.dofs_b])]
    # entry (j, j) of each column j
    diagonal = position[vec(build_nzidx_map(A_cpu, reshape(collect(axes(A_cpu, 2)), 1, :)))]
    return EvolutionLHSCache(on_architecture(arch, nonzeros(CᵀMC)[order]),
                             on_architecture(arch, nonzeros(CᵀK⁰C)[order]),
                             on_architecture(arch, Int32.(diagonal)), on_architecture(arch, nzidx),
                             on_architecture(arch, zeros(length(cache.w), size(cache.dofs_b, 2))),
                             Ref(θ))
end

struct EvolutionToolkit{A<:AbstractArchitecture, AC, L, CM, V, S<:IterativeSolverToolkit}
    arch::A
    cache::AC        # fe_data.cache on arch
    lhs::L           # EvolutionLHSCache
    C::CM            # b = C x + g, from the solution x on the free DOFs
    Cᵀ::CM
    g::V             # buoyancy at the Dirichlet DOFs, zero elsewhere
    f_bc_M::V        # -M g
    rhs_diff⁰::V     # -N² κᵥ ∂z(b) source term and -(Kₕ + Kᵥ⁰) g, for the background κᵥ
    rhs_diff::V      # rhs_diff⁰ plus the convection parameterization
    rhs_flux::V      # α F surface flux
    solver::S
end

function Base.summary(evolution::EvolutionToolkit)
    t = typeof(evolution)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, evolution::EvolutionToolkit)
    println(io, summary(evolution), ":")
    println(io, "├── arch: ", evolution.arch)
    println(io, "├── rhs_diff: ", summary(evolution.rhs_diff))
    println(io, "├── rhs_flux: ", summary(evolution.rhs_flux))
      print(io, "└── solver: ", summary(evolution.solver))
end

"""
    evolution_toolkit = EvolutionToolkit(arch, fe_data, params, forcings, ts; kwargs...)

Set up the toolkit for the buoyancy evolution equation

    μϱ (∂ₜb + u·∇b) = α²ε² [∇ₕ·(κₕ∇ₕb) + ∂z(κᵥ ∂z b)]

on the free DOFs of `fe_data.ch_b`.
"""
function EvolutionToolkit(arch::AbstractArchitecture,
                          fe_data::FEData,
                          params::Parameters,
                          forcings::Forcings,
                          ts::AbstractTimestepper;
                          atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=false)
    @info "Building evolution system..."

    ch_b = fe_data.ch_b
    is_free = trues(fe_data.nb)
    is_free[ch_b.prescribed_dofs] .= false
    C = constraint_matrix(ch_b, fe_data.nb, findall(is_free))
    g = apply!(zeros(fe_data.nb), ch_b)

    M  = build_M(fe_data)
    K⁰ = build_Kₕ(fe_data, forcings.κₕ)
    nonzeros(K⁰) .+= nonzeros(build_Kᵥ(fe_data, forcings.κᵥ))   # same sparsity pattern
    CᵀMC, CᵀK⁰C = C' * M * C, C' * K⁰ * C
    θ = evolution_parameter(params, ts)
    A = copy(CᵀMC)
    nonzeros(A) .= nonzeros(CᵀMC) .+ θ .* nonzeros(CᵀK⁰C)

    if use_diag_precond(arch, forcings, ts)
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $(size(A, 1)) DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end
    A   = on_architecture(arch, A)
    lhs = EvolutionLHSCache(fe_data, C, A, CᵀMC, CᵀK⁰C, θ)

    rhs_diff⁰ = on_architecture(arch, build_rhs_diff(params, fe_data, forcings.κᵥ) - K⁰ * g)
    rhs_flux  = on_architecture(arch, build_rhs_flux(params, forcings, fe_data))

    N  = size(A, 1)
    T  = eltype(A)
    y  = on_architecture(arch, zeros(T, N))
    workspace = Krylov.CgWorkspace(N, N, vector_type(arch, T))
    workspace.x .= zero(T)

    verbose_int = verbose ? 1 : 0
    kwargs_dict = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)
    solver = IterativeSolverToolkit(A, P, y, workspace, kwargs_dict, "Evolution")

    return EvolutionToolkit(arch, on_architecture(arch, fe_data.cache), lhs,
                            on_architecture(arch, C), on_architecture(arch, sparse(C')),
                            on_architecture(arch, g), on_architecture(arch, -(M * g)),
                            rhs_diff⁰, copy(rhs_diff⁰), rhs_flux, solver)
end

"""
    update_evolution_LHS!(evolution, params, forcings, ts, b)

Form the evolution matrix and its preconditioner for the timestep of `ts`, and, if
the convection parameterization is on, the matrix and `rhs_diff` for the buoyancy `b`.
"""
function update_evolution_LHS!(evolution::EvolutionToolkit, params::Parameters,
                               forcings::Forcings, ts::AbstractTimestepper, b::AbstractVector)
    lhs     = evolution.lhs
    conv_on = forcings.conv_param.is_on
    θ       = evolution_parameter(params, ts)
    (conv_on || θ != lhs.θ[]) || return evolution
    lhs.θ[] = θ
    A = evolution.solver.A
    nonzeros(A) .= lhs.M .+ θ .* lhs.K⁰
    if conv_on
        copyto!(evolution.rhs_diff, evolution.rhs_diff⁰)
        @ctime "  build convection" add_convection!(A, evolution.rhs_diff, lhs, evolution.cache,
                                                   evolution.g, params, forcings.conv_param, b, θ)
    end
    update_P!(evolution.solver.P, A, lhs.diagonal)
    return evolution
end

# LU is only worthwhile on the CPU when the LHS never changes
use_diag_precond(arch, forcings, ts) =
    typeof(arch) == GPU || forcings.conv_param.is_on || ts.adaptive

"""
    θ = evolution_parameter(params, ts)

Coefficient θ in `A = M + θ*(Kₕ + Kᵥ)`.
"""
evolution_parameter(p::Parameters, ts::AbstractTimestepper) =
    effective_timestep(ts) * p.α^2 * p.ε^2 / p.μϱ

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
    f = build_rhs_adv(cache, params, u, b, u_prev, b_prev, ts)

Assemble, on the architecture of `cache`, the advection right-hand side

    ∫ (((1 + ω) b - ω²/(1 + ω) b_prev) / a - Δt/a (u⋆·∇b⋆ + w⋆ N²)) φᵢ dΩ

where `ω = step_ratio(ts)`, `a = (1 + 2ω)/(1 + ω)` and `⋆` marks the extrapolation
`(1 + ω) b - ω b_prev`.
"""
function build_rhs_adv(cache::AssemblyCache, params::Parameters,
                       u::AbstractVector, b::AbstractVector,
                       u_prev::AbstractVector, b_prev::AbstractVector,
                       ts::AbstractTimestepper)
    ω = step_ratio(ts)
    a = (1 + 2ω) / (1 + ω)
    f = zero(b)
    rhs_adv_kernel!(get_backend(f))(f, cache, params.N²,
                                    (b, bp) -> ((1 + ω)*b - ω^2/(1 + ω)*bp) / a,
                                    (b, bp) -> (1 + ω)*b - ω*bp, (u, up) -> (1 + ω)*u - ω*up,
                                    1/a, ts.Δt[], u, b, u_prev, b_prev; ndrange=size(cache.dofs_b))
    return f
end

# Add `∫ (b_comb(b, b_prev) - Δt_fac Δt (u_eff·∇(b_eff) + w_eff N²)) φᵢ dΩ` to `f`,
# where `b_comb`, `b_eff` and `u_eff` combine the current and previous fields.
@kernel function rhs_adv_kernel!(f, cache, N², b_comb, b_eff, u_eff, Δt_fac, Δt,
                                 u, b, u_prev, b_prev)
    i, c = @index(Global, NTuple)
    J⁻ᵀ = cache.Jinv_t[c]
    fᵢ = 0.0
    for q in axes(cache.phi_b, 1)
        # b, b_prev, and the reference-space gradient of b_eff
        b_q = 0.0; bp_q = 0.0
        g1 = 0.0; g2 = 0.0; g3 = 0.0
        for k in axes(cache.phi_b, 2)
            d = cache.dofs_b[k, c]
            b_q  += cache.phi_b[q, k] * b[d]
            bp_q += cache.phi_b[q, k] * b_prev[d]
            beff  = b_eff(b[d], b_prev[d])
            g1 += cache.dphi_b[1, k, q] * beff
            g2 += cache.dphi_b[2, k, q] * beff
            g3 += cache.dphi_b[3, k, q] * beff
        end
        u1 = 0.0; u2 = 0.0; u3 = 0.0
        for j in axes(cache.phi_u, 2)
            φ = cache.phi_u[q, j]
            d1, d2, d3 = cache.dofs_u[3j - 2, c], cache.dofs_u[3j - 1, c], cache.dofs_u[3j, c]
            u1 += φ * u_eff(u[d1], u_prev[d1])
            u2 += φ * u_eff(u[d2], u_prev[d2])
            u3 += φ * u_eff(u[d3], u_prev[d3])
        end
        adv = u1*_∂ᵣ(J⁻ᵀ, 1, g1, g2, g3) + u2*_∂ᵣ(J⁻ᵀ, 2, g1, g2, g3) +
              u3*_∂ᵣ(J⁻ᵀ, 3, g1, g2, g3) + u3*N²
        rhs_q = (b_comb(b_q, bp_q) - Δt_fac * Δt * adv) * cache.w[q] * cache.detJ[c]
        fᵢ += rhs_q * cache.phi_b[q, i]
    end
    @atomic f[cache.dofs_b[i, c]] += fᵢ
end

####
#### Convection parameterization (κᵥ depends on ∂z b)
####

"""
    add_convection!(A, rhs_diff, lhs, cache, g, params, conv_param, b, θ)

Add the convective diffusivity `κᶜ`, evaluated from `α(N² + ∂z b)`, to the evolution
matrix and right-hand side: `θ ∫ κᶜ ∂zφᵢ ∂zφⱼ dΩ` to `A`, and
`-∫ N² κᶜ ∂zφᵢ dΩ - ∫ κᶜ ∂zφᵢ ∂z g dΩ` to `rhs_diff`, for the Dirichlet values `g`.
`A`, `rhs_diff`, `lhs`, `cache`, `g` and `b` are on the same architecture.
"""
function add_convection!(A, rhs_diff, lhs::EvolutionLHSCache, cache, g, params::Parameters,
                         conv_param::ConvectionParameterization, b, θ)
    backend = get_backend(lhs.κdΩ)
    nq, ncells = size(lhs.κdΩ)
    n_b = size(cache.dofs_b, 1)
    convective_diffusivity_kernel!(backend)(lhs.κdΩ, b, cache, conv_param, params.α, params.N²;
                                            ndrange=(nq, ncells))
    convection_matrix_kernel!(backend)(nonzeros(A), rhs_diff, lhs.nzidx, lhs.κdΩ, cache, g, θ;
                                       ndrange=(n_b, n_b, ncells))
    convection_source_kernel!(backend)(rhs_diff, lhs.κdΩ, cache, params.N²; ndrange=(n_b, ncells))
    synchronize(backend)
    return A
end

@kernel function convective_diffusivity_kernel!(κdΩ, b, cache, conv_param, α, N²)
    q, c = @index(Global, NTuple)
    ∂z_b = vertical_derivative(b, cache, q, c)
    κdΩ[q, c] = κ_conv_extra(conv_param, α * (N² + ∂z_b)) * cache.w[q] * cache.detJ[c]
end

@kernel function convection_matrix_kernel!(nzval, rhs_diff, nzidx, κdΩ, cache, g, θ)
    i, j, c = @index(Global, NTuple)
    J⁻ᵀ = cache.Jinv_t[c]
    Kᵢⱼ = 0.0
    for q in axes(κdΩ, 1)
        ∂zφᵢ = _∂ᵣ(J⁻ᵀ, 3, cache.dphi_b[1, i, q], cache.dphi_b[2, i, q], cache.dphi_b[3, i, q])
        ∂zφⱼ = _∂ᵣ(J⁻ᵀ, 3, cache.dphi_b[1, j, q], cache.dphi_b[2, j, q], cache.dphi_b[3, j, q])
        Kᵢⱼ += κdΩ[q, c] * ∂zφᵢ * ∂zφⱼ
    end
    k = nzidx[size(cache.dofs_b, 1)*(j - 1) + i, c]
    if k > 0
        @atomic nzval[k] += θ * Kᵢⱼ
    end
    @atomic rhs_diff[cache.dofs_b[i, c]] += -Kᵢⱼ * g[cache.dofs_b[j, c]]
end

@kernel function convection_source_kernel!(rhs_diff, κdΩ, cache, N²)
    i, c = @index(Global, NTuple)
    J⁻ᵀ = cache.Jinv_t[c]
    fᵢ = 0.0
    for q in axes(κdΩ, 1)
        ∂zφᵢ = _∂ᵣ(J⁻ᵀ, 3, cache.dphi_b[1, i, q], cache.dphi_b[2, i, q], cache.dphi_b[3, i, q])
        fᵢ -= N² * κdΩ[q, c] * ∂zφᵢ
    end
    @atomic rhs_diff[cache.dofs_b[i, c]] += fᵢ
end
