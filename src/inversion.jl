"""
    InversionLHSCache(fe_data, A)

Buffers for adding the eddy-viscous term to the condensed inversion matrix `A` in
place, on the architecture of `A`. Each velocity row of `fe_data.C_up` has at most
one entry, equal to 1.

- `nzval⁰`: `nonzeros(A)` without the viscous term
- `nzidx`: position in `nonzeros(A)` of entry `(i, j)` of the velocity block of cell
  `c` at `nzidx[n*(j-1)+i, c]`, with `n` velocity DOFs per cell, or 0 if the entry
  is eliminated
- `νdΩ`: `2α²ε² ν w |det J|` at each quadrature point of each cell
- `cache`: the geometry and reference basis gradients of `fe_data.cache`
"""
struct InversionLHSCache{V, I, M, C}
    nzval⁰::V
    nzidx::I
    νdΩ::M
    cache::C
end

function Base.summary(lhs::InversionLHSCache)
    t = typeof(lhs)
    return "$(parentmodule(t)).$(nameof(t))"
end

function InversionLHSCache(fe_data::FEData, A)
    arch  = architecture(A)
    cache = fe_data.cache
    rows, columns, _ = findnz(fe_data.C_up)
    free_dof = zeros(Int, size(fe_data.C_up, 1))
    free_dof[rows] .= columns
    position = nonzero_positions(A)
    nzidx = [k == 0 ? Int32(0) : Int32(position[k])
             for k in build_nzidx_map(on_architecture(CPU(), A), free_dof[cache.dofs_u])]
    arrays = (; cache.dofs_b, cache.dphi_b, cache.dphi_u, cache.Jinv_t, cache.J,
                cache.x0, cache.x_ref, cache.w, cache.detJ)
    return InversionLHSCache(copy(nonzeros(A)), on_architecture(arch, nzidx),
                             on_architecture(arch, zeros(length(cache.w), size(cache.dofs_u, 2))),
                             map(a -> on_architecture(arch, a), arrays))
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

    eddy_on = forcings.eddy_param.is_on
    A = on_architecture(arch, C' * build_A_inversion(fe_data, params, eddy_on ? 0.0 : forcings.ν) * C)
    lhs_cache = nothing
    if eddy_on
        lhs_cache = InversionLHSCache(fe_data, A)
        build_A_visc!(A, lhs_cache, params, forcings.eddy_param, on_architecture(arch, zeros(fe_data.nb)))
    end

    A_cpu = on_architecture(CPU(), A)
    if arch isa GPU
        P = BlockTriangularPreconditioner(arch, fe_data, A_cpu,
                CahouetChabardInverse(arch, fe_data, params, forcings, zeros(fe_data.nb)))
    else
        @warn "LU-factoring inversion matrix with $(size(A,1)) DOFs..."
        @time "lu(A_inversion)" P = lu(A_cpu)
    end

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
    build_A_visc!(A, lhs, params, eddy_param, b)

Set `A = A⁰ + ∫ 2α²ε² ν ε(u):ε(v) dΩ` in place, with the eddy viscosity `ν` evaluated
from `α(N² + ∂z b)`. `A`, `lhs` and `b` are on the same architecture, and `eddy_param.f`
must be callable there.
"""
function build_A_visc!(A, lhs::InversionLHSCache, params::Parameters,
                       eddy_param::EddyParameterization, b::AbstractVector)
    backend = get_backend(lhs.νdΩ)
    nq, ncells = size(lhs.νdΩ)
    n_nodes = size(lhs.cache.dphi_u, 2)
    eddy_viscosity_kernel!(backend)(lhs.νdΩ, b, lhs.cache, eddy_param, params.α, params.ε, params.N²;
                                    ndrange=(nq, ncells))
    copyto!(nonzeros(A), lhs.nzval⁰)
    viscous_matrix_kernel!(backend)(nonzeros(A), lhs.nzidx, lhs.νdΩ, lhs.cache;
                                    ndrange=(n_nodes, n_nodes, ncells))
    synchronize(backend)
    return A
end

@kernel function eddy_viscosity_kernel!(νdΩ, b, cache, eddy_param, α, ε, N²)
    q, c = @index(Global, NTuple)
    # reference-space gradient of b
    g1 = 0.0; g2 = 0.0; g3 = 0.0
    for i in axes(cache.dofs_b, 1)
        bᵢ = b[cache.dofs_b[i, c]]
        g1 += cache.dphi_b[1, i, q] * bᵢ
        g2 += cache.dphi_b[2, i, q] * bᵢ
        g3 += cache.dphi_b[3, i, q] * bᵢ
    end
    ∂z_b = _∂ᵣ(cache.Jinv_t[c], 3, g1, g2, g3)
    x = cache.x0[c] + cache.J[c] ⋅ cache.x_ref[q]
    ν = ν_eddy(eddy_param, eddy_param.f(x), α * (N² + ∂z_b))
    νdΩ[q, c] = 2α^2 * ε^2 * ν * cache.w[q] * cache.detJ[c]
end

@kernel function viscous_matrix_kernel!(nzval, nzidx, νdΩ, cache)
    m, n, c = @index(Global, NTuple)
    J⁻ᵀ = cache.Jinv_t[c]
    K = zero(Tensor{2, 3, Float64})
    for q in axes(νdΩ, 1)
        ∇φₘ = J⁻ᵀ ⋅ Vec{3}((cache.dphi_u[1, m, q], cache.dphi_u[2, m, q], cache.dphi_u[3, m, q]))
        ∇φₙ = J⁻ᵀ ⋅ Vec{3}((cache.dphi_u[1, n, q], cache.dphi_u[2, n, q], cache.dphi_u[3, n, q]))
        # for basis functions φₘ êᵣ and φₙ êₛ:  ε:ε = (δᵣₛ ∇φₘ·∇φₙ + ∂ₛφₘ ∂ᵣφₙ) / 2
        K += νdΩ[q, c] / 2 * ((∇φₘ ⋅ ∇φₙ) * one(K) + ∇φₙ ⊗ ∇φₘ)
    end
    n_u = 3 * size(cache.dphi_u, 2)
    for s in 1:3, r in 1:3
        k = nzidx[n_u*(3(n - 1) + s - 1) + 3(m - 1) + r, c]
        if k > 0
            @atomic nzval[k] += K[r, s]
        end
    end
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
