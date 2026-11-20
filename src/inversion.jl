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
"""
struct InversionLHSCache{V, I, M}
    nzval⁰::V
    nzidx::I
    νdΩ::M
end

function Base.summary(lhs::InversionLHSCache)
    t = typeof(lhs)
    return "$(parentmodule(t)).$(nameof(t))"
end

function InversionLHSCache(fe_data::FEData, A)
    arch  = architecture(A)
    cache = fe_data.cache
    free_u = free_index(fe_data.C_up)[fe_data.u_dof_indices]
    position = nonzero_positions(A)
    nzidx = [k == 0 ? Int32(0) : Int32(position[k])
             for k in build_nzidx_map(on_architecture(CPU(), A), free_u[cache.dofs_u])]
    return InversionLHSCache(copy(nonzeros(A)), on_architecture(arch, nzidx),
                             on_architecture(arch, zeros(length(cache.w), size(cache.dofs_u, 2))))
end

struct InversionToolkit{B, V, S<:IterativeSolverToolkit, AC, L, C}
    B::B        # maps the buoyancy DOFs to the right-hand side
    f_wind::V   # right-hand side from the wind stress
    solver::S
    cache::AC   # fe_data.cache on the solver's architecture
    lhs_cache::L   # InversionLHSCache if the eddy parameterization is on
    C_u::C      # maps the solution to the velocity DOFs
    C_p::C      # maps the solution to the pressure DOFs
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
    cache   = on_architecture(arch, fe_data.cache)
    b⁰      = on_architecture(arch, zeros(fe_data.nb))
    A = on_architecture(arch, C' * build_A_inversion(fe_data, params, eddy_on ? 0.0 : forcings.ν) * C)
    lhs_cache = nothing
    if eddy_on
        lhs_cache = InversionLHSCache(fe_data, A)
        build_A_visc!(A, lhs_cache, cache, params, forcings.eddy_param, b⁰)
    end

    if arch isa GPU
        P = BlockTriangularPreconditioner(arch, fe_data, A,
                CahouetChabardInverse(arch, fe_data, params, forcings, cache, b⁰))
    else
        @warn "LU-factoring inversion matrix with $(size(A,1)) DOFs..."
        @time "lu(A_inversion)" P = lu(A)
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

    return InversionToolkit(B, f_wind, solver, cache, lhs_cache,
                            on_architecture(arch, select_rows(C, fe_data.u_dof_indices)),
                            on_architecture(arch, select_rows(C, fe_data.p_dof_indices)))
end

"""
    invert!(inversion, b)

Solve the inversion for the buoyancy DOF vector `b` on the architecture of the
solver, leaving the solution on the free DOFs in `inversion.solver.x`.
"""
function invert!(inversion::InversionToolkit, b::AbstractVector)
    y = inversion.solver.y
    y .= inversion.f_wind
    mul!(y, inversion.B, b, 1.0, 1.0)
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
    A      = allocate_inversion_matrix(fe_data)
    cache  = fe_data.cache
    dofs_u = fe_data.u_dof_indices[cache.dofs_u]
    dofs_p = fe_data.p_dof_indices[cache.dofs_p]
    n_nodes, n_p = size(cache.phi_u, 2), size(cache.phi_p, 2)
    inversion_matrix_kernel!(get_backend(nonzeros(A)))(nonzeros(A), rowvals(A), SparseArrays.getcolptr(A),
                                                       dofs_u, dofs_p, cache, ν, params.f, params.α^2*params.ε^2;
                                                       ndrange=(n_nodes, n_nodes + n_p, size(dofs_u, 2)))
    return A
end

# Entries of cell c between velocity node a and velocity node b ≤ n_nodes, or pressure DOF b - n_nodes
@kernel function inversion_matrix_kernel!(nzval, rowval, colptr, dofs_u, dofs_p, cache, ν, f, α²ε²)
    a, b, c = @index(Global, NTuple)
    n_nodes = size(cache.phi_u, 2)
    J⁻ᵀ = cache.Jinv_t[c]
    if b ≤ n_nodes
        # viscous and Coriolis, for u = φ_b êₛ and v = φ_a êᵣ; (ẑ × u)·v = u₁v₂ - u₂v₁
        K = zero(Tensor{2, 3, Float64})
        for q in axes(cache.phi_u, 1)
            x   = cache.x0[c] + cache.J[c] ⋅ cache.x_ref[q]
            dΩ  = cache.w[q] * cache.detJ[c]
            ν_x = ν isa Function ? ν(x) : ν
            K  += 2α²ε² * ν_x * dΩ * strain_product(J⁻ᵀ, cache.dphi_u, a, b, q) +
                  f(x) * cache.phi_u[q, a] * cache.phi_u[q, b] * dΩ *
                  Tensor{2, 3, Float64}((r, s) -> (r == 2 && s == 1) - (r == 1 && s == 2))
        end
        for s in 1:3, r in 1:3
            add_entry!(nzval, rowval, colptr, dofs_u[3(a - 1) + r, c], dofs_u[3(b - 1) + s, c], K[r, s])
        end
    else
        # pressure gradient -(∇·v) p and divergence q (∇·u), for v = φ_a êᵣ and p = ψ_(b - n_nodes)
        for r in 1:3
            G = 0.0
            for q in axes(cache.phi_u, 1)
                G += _∂ᵣ(J⁻ᵀ, r, cache.dphi_u[1, a, q], cache.dphi_u[2, a, q], cache.dphi_u[3, a, q]) *
                     cache.phi_p[q, b - n_nodes] * cache.w[q] * cache.detJ[c]
            end
            add_entry!(nzval, rowval, colptr, dofs_u[3(a - 1) + r, c], dofs_p[b - n_nodes, c], -G)
            add_entry!(nzval, rowval, colptr, dofs_p[b - n_nodes, c], dofs_u[3(a - 1) + r, c], G)
        end
    end
end

"""
    S = strain_product(J⁻ᵀ, dphi_u, m, n, q)

`ε(φₙ êₛ):ε(φₘ êᵣ)` as `S[r, s]`, at quadrature point `q` of a cell with inverse
transposed Jacobian `J⁻ᵀ`.
"""
@inline function strain_product(J⁻ᵀ, dphi_u, m, n, q)
    ∇φₘ = J⁻ᵀ ⋅ Vec{3}((dphi_u[1, m, q], dphi_u[2, m, q], dphi_u[3, m, q]))
    ∇φₙ = J⁻ᵀ ⋅ Vec{3}((dphi_u[1, n, q], dphi_u[2, n, q], dphi_u[3, n, q]))
    return ((∇φₘ ⋅ ∇φₙ) * one(Tensor{2, 3, Float64}) + ∇φₙ ⊗ ∇φₘ) / 2
end

"""
    add_entry!(nzval, rowval, colptr, row, col, v)

Add `v` to entry `(row, col)` of the CSC matrix with the given arrays, which stores it.
"""
@inline function add_entry!(nzval, rowval, colptr, row, col, v)
    k = searchsortedfirst(rowval, row, colptr[col], colptr[col + 1] - 1, Base.Order.Forward)
    @atomic nzval[k] += v
    return nothing
end

"""
    build_A_visc!(A, lhs, cache, params, eddy_param, b)

Set `A = A⁰ + ∫ 2α²ε² ν ε(u):ε(v) dΩ` in place, with the eddy viscosity `ν` evaluated
from `α(N² + ∂z b)`. `A`, `lhs`, `cache` and `b` are on the same architecture, and
`eddy_param.f` must be callable there.
"""
function build_A_visc!(A, lhs::InversionLHSCache, cache::AssemblyCache, params::Parameters,
                       eddy_param::EddyParameterization, b::AbstractVector)
    backend = get_backend(lhs.νdΩ)
    nq, ncells = size(lhs.νdΩ)
    n_nodes = size(cache.dphi_u, 2)
    eddy_viscosity_kernel!(backend)(lhs.νdΩ, b, cache, eddy_param, params.α, params.ε, params.N²;
                                    ndrange=(nq, ncells))
    copyto!(nonzeros(A), lhs.nzval⁰)
    viscous_matrix_kernel!(backend)(nonzeros(A), lhs.nzidx, lhs.νdΩ, cache;
                                    ndrange=(n_nodes, n_nodes, ncells))
    synchronize(backend)
    return A
end

@kernel function eddy_viscosity_kernel!(νdΩ, b, cache, eddy_param, α, ε, N²)
    q, c = @index(Global, NTuple)
    ∂z_b = vertical_derivative(b, cache, q, c)
    x = cache.x0[c] + cache.J[c] ⋅ cache.x_ref[q]
    ν = ν_eddy(eddy_param, eddy_param.f(x), α * (N² + ∂z_b))
    νdΩ[q, c] = 2α^2 * ε^2 * ν * cache.w[q] * cache.detJ[c]
end

@kernel function viscous_matrix_kernel!(nzval, nzidx, νdΩ, cache)
    m, n, c = @index(Global, NTuple)
    J⁻ᵀ = cache.Jinv_t[c]
    K = zero(Tensor{2, 3, Float64})
    for q in axes(νdΩ, 1)
        K += νdΩ[q, c] * strain_product(J⁻ᵀ, cache.dphi_u, m, n, q)
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
