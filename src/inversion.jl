"""
    InversionLHSCache(A⁰, fe_data)

Buffers for rebuilding the condensed inversion matrix in place when only the
viscosity changes.

- `A⁰`: uncondensed matrix without the viscous term
- `A`: `A⁰` plus the current viscous term
- `A_cond`: condensed matrix `CᵀAC`
- `nzidx_up`: index into `A.nzval` of every entry of each cell's velocity block
- `sources`, `destinations`, `coefficients`: `A.nzval[sources[t]]` contributes
  `coefficients[t]` times its value to `A_cond.nzval[destinations[t]]`
- `gpu_perm`: cache for [`update_A!`](@ref) on the GPU
"""
struct InversionLHSCache
    A⁰::SparseMatrixCSC{Float64, Int}
    A::SparseMatrixCSC{Float64, Int}
    A_cond::SparseMatrixCSC{Float64, Int}
    nzidx_up::Matrix{Int}
    sources::Vector{Int}
    destinations::Vector{Int}
    coefficients::Vector{Float64}
    gpu_perm::Ref{Any}
end

function Base.summary(lhs::InversionLHSCache)
    t = typeof(lhs)
    return "$(parentmodule(t)).$(nameof(t))"
end

function InversionLHSCache(A⁰::SparseMatrixCSC, fe_data::FEData)
    sources, destinations, coefficients, A_cond = _condensation_map(A⁰, fe_data.C_up)
    nzidx_up = build_nzidx_map(A⁰, fe_data.cache.dofs_u)
    return InversionLHSCache(A⁰, copy(A⁰), A_cond, nzidx_up,
                             sources, destinations, coefficients, Ref{Any}(nothing))
end

"""
    sources, destinations, coefficients, A_cond = _condensation_map(A, C)

Map from the nonzeros of `A` to those of `CᵀAC`: entry `A[p, q]` contributes
`C[p, i] C[q, j]` times its value to entry `(i, j)`. `A_cond` is a zero matrix
with the sparsity pattern of `CᵀAC` for any values of `A`.
"""
function _condensation_map(A::SparseMatrixCSC, C::SparseMatrixCSC)
    Cᵀ = sparse(C')   # column p of Cᵀ is row p of C
    rows_C, vals_C = rowvals(Cᵀ), nonzeros(Cᵀ)

    Is = Int[]; Js = Int[]; sources = Int[]; coefficients = Float64[]
    for q in 1:size(A, 2), k in nzrange(A, q)
        p = rowvals(A)[k]
        for a in nzrange(Cᵀ, p), b in nzrange(Cᵀ, q)
            push!(Is, rows_C[a]); push!(Js, rows_C[b])
            push!(sources, k)
            push!(coefficients, vals_C[a] * vals_C[b])
        end
    end

    N_free = size(C, 2)
    A_cond = sparse(Is, Js, zeros(length(Is)), N_free, N_free)
    destinations = map(Is, Js) do i, j
        column = nzrange(A_cond, j)
        searchsortedfirst(rowvals(A_cond), i, first(column), last(column), Base.Order.Forward)
    end
    return sources, destinations, coefficients, A_cond
end

"""
    refresh_A_cond!(lhs)

Recompute `lhs.A_cond` from the current `lhs.A`.
"""
function refresh_A_cond!(lhs::InversionLHSCache)
    A, A_cond = lhs.A, lhs.A_cond
    fill!(A_cond.nzval, 0.0)
    @inbounds for t in eachindex(lhs.sources)
        A_cond.nzval[lhs.destinations[t]] += lhs.coefficients[t] * A.nzval[lhs.sources[t]]
    end
    return A_cond
end

struct InversionToolkit{B, V, S<:IterativeSolverToolkit, L}
    B::B        # maps the buoyancy DOFs to the right-hand side
    f_wind::V   # right-hand side from the wind stress
    b_arch::V   # buoyancy on the solver's architecture
    solver::S
    lhs_cache::L              # InversionLHSCache if the eddy parameterization is on
    x_full::Vector{Float64}   # solution on all DOFs of dh_up
end

function Base.summary(inversion::InversionToolkit)
    t = typeof(inversion)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, inversion::InversionToolkit)
    println(io, summary(inversion), ":")
    println(io, "├── B: ", summary(inversion.B))
    println(io, "├── f_wind: ", summary(inversion.f_wind))
      print(io, "└── solver: ", summary(inversion.solver))
end

"""
    inversion = InversionToolkit(arch, fe_data, params, forcings; kwargs...)

Set up the solver for the PG inversion

    -f(ẑ×u) + α²ε² ∇·(2ν ε(u)) - ∇p = (1/α) b ẑ
    ∇·u = 0

on the free DOFs of `fe_data.dh_up`. The system is LU-factorized on the CPU. On
the GPU it is solved by FGMRES with a [`BlockTriangularPreconditioner`](@ref),
and the keyword arguments are those of `Krylov.fgmres`.
"""
function InversionToolkit(arch::AbstractArchitecture,
                          fe_data::FEData,
                          params::Parameters,
                          forcings::Forcings;
                          atol=1e-6, rtol=1e-6, itmax=0,
                          memory=100,   # FGMRES(20) stagnates on warm-started solves
                          history=true, verbose=false, restart=true)
    @info "Building inversion system..."

    C      = fe_data.C_up
    B      = C' * build_B_inversion(fe_data, params)
    f_wind = C' * build_f_wind(fe_data, params, forcings)

    if forcings.eddy_param.is_on
        lhs_cache = InversionLHSCache(build_A_inversion(fe_data, params, 0.0), fe_data)
        build_A_visc!(lhs_cache.A, lhs_cache.A⁰, fe_data, params,
                      forcings.eddy_param, zeros(fe_data.nb), lhs_cache.nzidx_up)
        A = refresh_A_cond!(lhs_cache)
    else
        lhs_cache = nothing
        # unlike Ferrite's `apply!`, this eliminates the constrained DOFs
        A = C' * build_A_inversion(fe_data, params, forcings.ν) * C
    end

    if arch isa GPU
        P = BlockTriangularPreconditioner(arch, fe_data, params, forcings, A, zeros(fe_data.nb))
    else
        @warn "LU-factoring inversion matrix with $(size(A,1)) DOFs..."
        @time "lu(A_inversion)" P = lu(A)
    end

    A      = on_architecture(arch, A)
    B      = on_architecture(arch, B)
    f_wind = on_architecture(arch, f_wind)
    print_memory_status(arch)

    N  = size(A, 1)
    T  = eltype(A)
    y  = on_architecture(arch, zeros(T, N))
    workspace = Krylov.FgmresWorkspace(N, N, vector_type(arch, T); memory)
    workspace.x .= zero(T)

    kwargs_dict = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history,
                       :verbose=>Int(verbose), :restart=>restart)
    solver = IterativeSolverToolkit(A, P, y, workspace, kwargs_dict, "Inversion")

    return InversionToolkit(B, f_wind, on_architecture(arch, zeros(fe_data.nb)), solver,
                            lhs_cache, zeros(ndofs(fe_data.dh_up)))
end

"""
    invert!(inversion, b)

Solve the inversion for the buoyancy DOF vector `b`, leaving the solution on the
free DOFs in `inversion.solver.x`.
"""
function invert!(inversion::InversionToolkit, b::AbstractVector)
    y = inversion.solver.y
    copyto!(inversion.b_arch, b)
    y .= inversion.f_wind
    mul!(y, inversion.B, inversion.b_arch, 1.0, 1.0)
    iterative_solve!(inversion.solver)
    return inversion
end

####
#### Matrix and vector builders
####

"""
    A = build_A_inversion(fe_data, params, ν)

Assemble the LHS matrix for the inversion problem:

    A[(u,v)] = ∫ 2α²ε² ν ε(u):ε(v) dΩ  (viscous)
             + ∫ f (ẑ×u)·v dΩ            (Coriolis)
             - ∫ (∇·v) p dΩ              (pressure gradient)
             + ∫ q (∇·u) dΩ              (divergence-free)
"""
function build_A_inversion(fe_data::FEData, params::Parameters, ν)
    dh_up = fe_data.dh_up
    cv_u, cv_p, _ = make_cell_values(fe_data)
    n_u   = getnbasefunctions(cv_u)
    n_p   = getnbasefunctions(cv_p)
    n_loc = n_u + n_p
    α²ε²  = params.α^2 * params.ε^2
    f_cor = params.f   # Coriolis parameter (function of x)

    A   = allocate_inversion_matrix(fe_data)
    asm = start_assemble(A)
    Ae  = zeros(n_loc, n_loc)

    for cc in CellIterator(dh_up)
        reinit!(cv_u, cc)
        reinit!(cv_p, cc)
        coords = getcoordinates(cc)
        fill!(Ae, 0.0)

        for q in 1:getnquadpoints(cv_u)
            x  = spatial_coordinate(cv_u, q, coords)
            ν_q = ν isa Function ? ν(x) : ν
            f_q = f_cor(x)
            dΩ  = getdetJdV(cv_u, q)

            for i in 1:n_u
                ε_i    = symmetric(shape_gradient(cv_u, q, i))
                φᵤ_i   = shape_value(cv_u, q, i)
                div_i  = tr(shape_gradient(cv_u, q, i))

                for j in 1:n_u
                    φⱼ = shape_value(cv_u, q, j)
                    # viscous: 2α²ε² ν ε(u):ε(v)
                    visc = 2α²ε² * ν_q * dcontract(ε_i, symmetric(shape_gradient(cv_u, q, j)))
                    # Coriolis: f (ẑ×u)·v,  ẑ×u = (-u₂, u₁, 0)
                    cori = f_q * (-φⱼ[2] * φᵤ_i[1] + φⱼ[1] * φᵤ_i[2])
                    Ae[i, j] += (visc + cori) * dΩ
                end

                for j in 1:n_p
                    # pressure gradient: -(∇·v) p
                    Ae[i, n_u + j] -= div_i * shape_value(cv_p, q, j) * dΩ
                end
            end

            for i in 1:n_p
                φ_p_i = shape_value(cv_p, q, i)
                for j in 1:n_u
                    # divergence-free: q (∇·u)
                    Ae[n_u + i, j] += φ_p_i * tr(shape_gradient(cv_u, q, j)) * dΩ
                end
            end
        end

        assemble!(asm, celldofs(cc), Ae)
    end
    return A
end

"""
    build_A_visc!(A, A⁰, fe_data, params, eddy_param, b, nzidx_up)

Set `A = A⁰ + ∫ 2α²ε² ν ε(u):ε(v) dΩ`, with the eddy viscosity `ν` evaluated from
`α(N² + ∂z b)`. `nzidx_up` is `build_nzidx_map(A, fe_data.cache.dofs_u)`.
"""
function build_A_visc!(A::SparseMatrixCSC, A⁰::SparseMatrixCSC,
                       fe_data::FEData, params::Parameters,
                       eddy_param::EddyParameterization, b::AbstractVector,
                       nzidx_up::Matrix{Int})
    cache = fe_data.cache
    nq   = length(cache.w)
    n_nodes = size(cache.dphi_u, 2)   # velocity nodes per cell
    n_u  = 3 * n_nodes
    n_b  = size(cache.phi_b, 2)
    ncells = length(cache.detJ)
    α²ε² = params.α^2 * params.ε^2
    α    = params.α
    N²   = params.N²

    A.nzval .= A⁰.nzval

    b_cell = zeros(n_b)
    ∇φ = zeros(3, n_nodes)
    Ke = zeros(n_u, n_u)

    @inbounds for c in 1:ncells
        for i in 1:n_b
            b_cell[i] = b[cache.dofs_b[i, c]]
        end
        J⁻ᵀ = cache.Jinv_t[c]
        dJ = cache.detJ[c]
        x0 = cache.x0[c]
        J  = cache.J[c]
        fill!(Ke, 0.0)

        for q in 1:nq
            # reference-space gradient of b
            g1 = 0.0; g2 = 0.0; g3 = 0.0
            for i in 1:n_b
                g1 += cache.dphi_b[1, i, q] * b_cell[i]
                g2 += cache.dphi_b[2, i, q] * b_cell[i]
                g3 += cache.dphi_b[3, i, q] * b_cell[i]
            end
            ∂z_b = _∂ᵣ(J⁻ᵀ, 3, g1, g2, g3)
            x    = x0 + J ⋅ cache.x_ref[q]
            ν    = ν_eddy(eddy_param, eddy_param.f(x), α * (N² + ∂z_b))
            νdΩ  = 2α²ε² * ν * cache.w[q] * dJ

            for m in 1:n_nodes, r in 1:3
                ∇φ[r, m] = _∂ᵣ(J⁻ᵀ, r, cache.dphi_u[1, m, q], cache.dphi_u[2, m, q], cache.dphi_u[3, m, q])
            end

            # for basis functions φₘ êᵣ and φₙ êₛ:  ε:ε = (δᵣₛ ∇φₘ·∇φₙ + ∂ₛφₘ ∂ᵣφₙ) / 2
            for n in 1:n_nodes, m in 1:n_nodes
                ∇φₘ∇φₙ = ∇φ[1, m]*∇φ[1, n] + ∇φ[2, m]*∇φ[2, n] + ∇φ[3, m]*∇φ[3, n]
                for s in 1:3, r in 1:3
                    i = 3*(m - 1) + r
                    j = 3*(n - 1) + s
                    Ke[i, j] += νdΩ * ((r == s ? ∇φₘ∇φₙ : 0.0) + ∇φ[s, m] * ∇φ[r, n]) / 2
                end
            end
        end

        for j in 1:n_u, i in 1:n_u
            A.nzval[nzidx_up[n_u*(j - 1) + i, c]] += Ke[i, j]
        end
    end
    return A
end

"""
    B = build_B_inversion(fe_data, params)

Assemble the buoyancy-to-velocity coupling matrix:

    B[u_i, b_j] = ∫ (1/α) φ_b_j (ẑ·φᵤ_i) dΩ

`B` is `N_up × nb`, with `N_up = ndofs(dh_up)`.
"""
function build_B_inversion(fe_data::FEData, params::Parameters)
    dh_up = fe_data.dh_up
    dh_b  = fe_data.dh_b
    cv_u, _, cv_b = make_cell_values(fe_data)
    n_u = getnbasefunctions(cv_u)
    n_b = getnbasefunctions(cv_b)
    α   = params.α
    N_up = ndofs(dh_up)
    nb   = fe_data.nb

    rows = Int[]; cols = Int[]; vals = Float64[]

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_u, cc_up)
        reinit!(cv_b, cc_b)
        dofs_up = celldofs(cc_up)
        dofs_b  = celldofs(cc_b)
        Be = zeros(n_u, n_b)

        for q in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q)
            for i in 1:n_u
                z_comp = shape_value(cv_u, q, i)[3]   # ẑ·φᵤ_i
                iszero(z_comp) && continue
                for j in 1:n_b
                    Be[i, j] += (1/α) * shape_value(cv_b, q, j) * z_comp * dΩ
                end
            end
        end

        u_range = dof_range(dh_up, :u)
        for (li, gi) in enumerate(dofs_up[u_range])
            for (lj, gj) in enumerate(dofs_b)
                v = Be[li, lj]
                iszero(v) && continue
                push!(rows, gi); push!(cols, gj); push!(vals, v)
            end
        end
    end

    return sparse(rows, cols, vals, N_up, nb)
end

"""
    f = build_f_wind(fe_data, params, forcings)

Assemble the wind-stress right-hand side over the surface `Γ`:

    f[u_i] = ∫_Γ α (τˣ (x̂·φᵤ_i) + τʸ (ŷ·φᵤ_i)) dΓ
"""
function build_f_wind(fe_data::FEData, params::Parameters, forcings::Forcings)
    dh_up     = fe_data.dh_up
    fv_u, _   = make_facet_values(fe_data)
    n_u       = getnbasefunctions(fv_u)
    α         = params.α
    τˣ        = forcings.τˣ
    τʸ        = forcings.τʸ
    facetset  = fe_data.mesh.grid.facetsets[fe_data.mesh.surface_tag]
    u_range   = dof_range(dh_up, :u)

    f  = zeros(ndofs(dh_up))
    fₑ = zeros(n_u)

    for fc in FacetIterator(dh_up, facetset)
        reinit!(fv_u, fc)
        coords = getcoordinates(fc)
        fill!(fₑ, 0.0)
        for q in 1:getnquadpoints(fv_u)
            x  = spatial_coordinate(fv_u, q, coords)
            dΓ = getdetJdV(fv_u, q)
            for i in 1:n_u
                φᵤ_i = shape_value(fv_u, q, i)
                fₑ[i] += α * (τˣ(x) * φᵤ_i[1] + τʸ(x) * φᵤ_i[2]) * dΓ
            end
        end
        f[celldofs(fc)[u_range]] .+= fₑ
    end
    return f
end
