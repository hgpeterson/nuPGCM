"""
    InversionLHSCache

Precomputed data for allocation-free rebuilds of the condensed inversion LHS
`A_cond = C'AC + D` when only the viscous block of the uncondensed `A`
changes (the eddy-viscosity parameterization: `ν` depends on `b`, everything
else is fixed). Analogous to `EvolutionLHSCache` for the buoyancy system, but
`condense_system`'s `C'AC` product (unlike Ferrite's symmetric-only `apply!`)
is not linear in a way that preserves a *fixed* sparsity pattern from one
`A` to the next: coincidental numeric cancellation can drop an entry from
`C'AC`'s realized pattern for one `ν` field but not another. `M_pattern` is
therefore built *structurally* — the union of every `(i,j)` destination that
any nonzero of `A` could possibly reach through `C` — so it is guaranteed to
contain every entry any `ν` field can produce; see `scratch/verify_condense_map.jl`
for the numerical check against `condense_system`'s brute-force output.

Refresh (`refresh_A_cond!`) is then a sparse scatter: for each nonzero of the
(rebuilt) uncondensed `A` at linear index `k`, add `coeffs[t] * A.nzval[k]`
to `A_cond.nzval[dests[t]]` for every `t` with `ks[t] == k`, then add the
constrained-row mean-|diagonal| `D` at `diag_dests`. No new sparse matrix is
allocated.

Fields:
- `A⁰`: static (ν-independent) uncondensed inversion matrix
- `A`: reusable uncondensed buffer (`A⁰` + current viscous block)
- `A_cond`: reusable condensed buffer, laid out on `M_pattern`
- `g`: full-length constraint-inhomogeneity vector (for `f_bc = C'(-A g)`)
- `nzidx_up`: local u-u block → `A.nzval` index map for [`build_A_visc!`](@ref);
  depends on the (uncondensed) inversion pattern, hence lives here rather than
  in the matrix-independent [`AssemblyCache`](@ref)
- `ks`, `dests`, `coeffs`: scatter map from `A.nzval` into `A_cond.nzval`
- `diag_dests`: positions of the constrained-row entries of `D` in `A_cond.nzval`
- `gpu_perm`: lazily-populated, architecture-specific scratch cache for
  in-place `solver.A` updates (see [`update_A!`](@ref) and
  `ext/nuPGCMCUDAExt.jl`); unused (stays `nothing`) on CPU
"""
struct InversionLHSCache
    A⁰::SparseMatrixCSC{Float64, Int}
    A::SparseMatrixCSC{Float64, Int}
    A_cond::SparseMatrixCSC{Float64, Int}
    g::Vector{Float64}
    nzidx_up::Matrix{Int}
    ks::Vector{Int}
    dests::Vector{Int}
    coeffs::Vector{Float64}
    diag_dests::Vector{Int}
    gpu_perm::Ref{Any}
end

function Base.summary(lhs::InversionLHSCache)
    t = typeof(lhs)
    return "$(parentmodule(t)).$(nameof(t))"
end

"""
    ks, dests, coeffs, diag_dests, M_pattern = _condense_scatter_map(A, C, ch)

Build the scatter map described in [`InversionLHSCache`](@ref): for each
nonzero `A[p,q]` (CSC linear index `k`) and each pair of nonzeros `C[p,i]`,
`C[q,j]`, register a contribution `C[p,i]*C[q,j] * A.nzval[k]` to destination
`(i,j)`. `M_pattern` is the structural union of all such `(i,j)` (values are
placeholders; only the sparsity layout is used), together with the
constrained diagonal `(cdof, cdof)` entries for `D`.
"""
function _condense_scatter_map(A::SparseMatrixCSC, C::SparseMatrixCSC, ch::ConstraintHandler)
    N = size(A, 1)

    # row-wise nonzeros of C: Crows[p] = [(i, C[p,i]) for i with C[p,i] != 0]
    Crows    = [Int[] for _ in 1:N]
    Crowsval = [Float64[] for _ in 1:N]
    for i in 1:N
        for idx in C.colptr[i]:(C.colptr[i+1]-1)
            p = C.rowval[idx]
            push!(Crows[p], i)
            push!(Crowsval[p], C.nzval[idx])
        end
    end

    Is = Int[]; Js = Int[]; Ks = Int[]; Coeffs = Float64[]
    for q in 1:N
        for idx in A.colptr[q]:(A.colptr[q+1]-1)
            p = A.rowval[idx]
            for (ii, ci) in zip(Crows[p], Crowsval[p])
                for (jj, cj) in zip(Crows[q], Crowsval[q])
                    coeff = ci * cj
                    coeff == 0.0 && continue
                    push!(Is, ii); push!(Js, jj); push!(Ks, idx); push!(Coeffs, coeff)
                end
            end
        end
    end
    # register D's constrained-diagonal entries as structural nonzeros too
    n_edge = length(Is)
    for cdof in ch.prescribed_dofs
        push!(Is, cdof); push!(Js, cdof)
    end

    # structural pattern: boolean OR combiner, so no candidate is dropped to
    # an accidental zero the way `+`-accumulation in a real C'AC product could
    M_pattern = sparse(Is, Js, trues(length(Is)), N, N, |)
    rowval_M  = M_pattern.rowval
    colptr_M  = M_pattern.colptr

    _dest(ii, jj) = begin
        rng = colptr_M[jj]:(colptr_M[jj+1]-1)
        pos = searchsortedfirst(rowval_M, ii, first(rng), last(rng), Base.Order.Forward)
        @assert pos <= last(rng) && rowval_M[pos] == ii "missing structural pattern entry ($ii,$jj)"
        pos
    end

    ks     = Vector{Int}(undef, n_edge)
    dests  = Vector{Int}(undef, n_edge)
    coeffs = Vector{Float64}(undef, n_edge)
    for t in 1:n_edge
        ks[t]     = Ks[t]
        coeffs[t] = Coeffs[t]
        dests[t]  = _dest(Is[t], Js[t])
    end

    diag_dests = [_dest(cdof, cdof) for cdof in ch.prescribed_dofs]

    return ks, dests, coeffs, diag_dests, M_pattern
end

function InversionLHSCache(A⁰::SparseMatrixCSC, fe_data::FEData)
    ch, C = fe_data.ch_up, fe_data.C_up
    ks, dests, coeffs, diag_dests, M_pattern = _condense_scatter_map(A⁰, C, ch)
    A_cond = SparseMatrixCSC(M_pattern.m, M_pattern.n, M_pattern.colptr,
                             M_pattern.rowval, zeros(length(M_pattern.nzval)))
    g = zeros(size(A⁰, 1))
    g[ch.prescribed_dofs] .= ch.inhomogeneities
    nzidx_up = build_nzidx_map(A⁰, fe_data.cache.dofs_u)
    return InversionLHSCache(A⁰, copy(A⁰), A_cond, g, nzidx_up,
                             ks, dests, coeffs, diag_dests, Ref{Any}(nothing))
end

"""
    A_cond, f_bc = refresh_A_cond!(lhs, ch)

Recombine `lhs.A_cond` from the current `lhs.A` (call [`build_A_visc!`](@ref)
first to refresh its viscous block) via the cached scatter map — no new
sparse matrix allocated. Returns the condensed matrix and the RHS correction
`f_bc = C'(-A g)` (zero for the homogeneous velocity BCs this model uses, but
computed in general).
"""
function refresh_A_cond!(lhs::InversionLHSCache, ch::ConstraintHandler)
    A, A_cond = lhs.A, lhs.A_cond
    N = size(A, 1)

    fill!(A_cond.nzval, 0.0)
    @inbounds for t in eachindex(lhs.ks)
        A_cond.nzval[lhs.dests[t]] += lhs.coeffs[t] * A.nzval[lhs.ks[t]]
    end
    md = sum(abs, diag(A)) / N
    @inbounds for pos in lhs.diag_dests
        A_cond.nzval[pos] += md
    end

    f_bc = -(A * lhs.g)
    _condense_rhs!(f_bc, ch)
    return A_cond, f_bc
end

struct InversionToolkit{B, V, CUP, S<:IterativeSolverToolkit, L}
    B::B       # RHS coupling matrix (N_up × nb): maps buoyancy DOFs to (u,p) DOFs
    f_wind::V  # RHS from wind-stress surface integral
    f_bc::V    # RHS correction for inhomogeneous BCs (0 for homogeneous)
    ch_up::CUP # ConstraintHandler for setting constrained DOF values in invert!
    solver::S
    lhs_cache::L   # InversionLHSCache when eddy_param is on, `nothing` otherwise
end

function Base.summary(inv::InversionToolkit)
    t = typeof(inv)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, inv::InversionToolkit)
    println(io, summary(inv), ":")
    println(io, "├── B: ", summary(inv.B))
    println(io, "├── f_wind: ", summary(inv.f_wind))
    println(io, "├── f_bc: ", summary(inv.f_bc))
      print(io, "└── solver: ", summary(inv.solver))
end

"""
    inversion_toolkit = InversionToolkit(arch, fe_data, params, forcings; kwargs...)

Set up the toolkit for inverting the steady PG momentum equations

    -f(ẑ×u) + α²ε² ∇·(2ν ε(u)) - ∇p = (1/α) b ẑ
    ∇·u = 0
"""
function InversionToolkit(arch::AbstractArchitecture,
                          fe_data::FEData,
                          params::Parameters,
                          forcings::Forcings;
                          atol=1e-6, rtol=1e-6, itmax=0,
                          memory=20, history=true, verbose=false, restart=true)
    @info "Building inversion system..."

    B      = build_B_inversion(fe_data, params)
    f_wind = build_f_wind(fe_data, params, forcings)

    eddy_on = forcings.eddy_param.is_on
    lhs_cache = nothing
    if eddy_on
        # A⁰ (ν-independent part) + viscous block from the initial b = 0, then
        # pre-build the condensation scatter map; _update_eddy_A! reuses both
        # for every subsequent rebuild (only the viscous nzval and the
        # condensed nzval it feeds are ever touched again).
        A⁰ = build_A_inversion_static(fe_data, params)
        lhs_cache = InversionLHSCache(A⁰, fe_data)
        build_A_visc!(lhs_cache.A, lhs_cache.A⁰, fe_data, params,
                      forcings.eddy_param, zeros(fe_data.nb), lhs_cache.nzidx_up)
        A, f_bc = refresh_A_cond!(lhs_cache, fe_data.ch_up)
    else
        A = build_A_inversion(fe_data, params, forcings.ν)
        # apply BCs: condense A, compute f_bc correction for inhomogeneous BC values
        # (not Ferrite's apply!, which corrupts non-symmetric matrices; see condense_system)
        A, f_bc = condense_system(A, fe_data.ch_up, fe_data.C_up)
    end

    # preconditioner
    if typeof(arch) == GPU || eddy_on
        p, t = get_p_t(fe_data.mesh)
        edges, _, _ = all_edges(t)
        hs = sort([norm(p[edges[i,1],:] - p[edges[i,2],:]) for i in axes(edges,1)])
        h  = hs[length(hs) ÷ 2]
        P  = Diagonal(on_architecture(arch, fill(1/h^3, size(A,1))))
    else
        @warn "LU-factoring inversion matrix with $(size(A,1)) DOFs..."
        @time "lu(A_inversion)" P = lu(A)
    end

    A      = on_architecture(arch, A)
    B      = on_architecture(arch, B)
    f_wind = on_architecture(arch, f_wind)
    f_bc   = on_architecture(arch, f_bc)
    print_memory_status(arch)

    N  = size(A, 1)
    T  = eltype(A)
    y  = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    workspace = Krylov.GmresWorkspace(N, N, VT; memory)
    workspace.x .= zero(T)

    verbose_int = verbose ? 1 : 0
    kwargs_dict = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax,
                       :history=>history, :verbose=>verbose_int, :restart=>restart)
    solver = IterativeSolverToolkit(A, P, y, workspace, kwargs_dict, "Inversion")

    return InversionToolkit(B, f_wind, f_bc, fe_data.ch_up, solver, lhs_cache)
end

"""
    invert!(inv_tk, b_vec)

Solve the inversion system given buoyancy DOF vector `b_vec` (length nb).
The combined (u,p) solution is stored in `inv_tk.solver.x`.
"""
function invert!(inv_tk::InversionToolkit, b_vec::AbstractVector)
    arch = architecture(inv_tk.solver.A)
    y = on_architecture(CPU(), inv_tk.B) * b_vec .+
        on_architecture(CPU(), inv_tk.f_wind) .+
        on_architecture(CPU(), inv_tk.f_bc)
    _condense_rhs!(y, inv_tk.ch_up)  # merge mirror rows into image rows, zero all constrained DOFs
    inv_tk.solver.y .= on_architecture(arch, y)
    iterative_solve!(inv_tk.solver)
    return inv_tk
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
    A⁰ = build_A_inversion_static(fe_data, params)

Assemble the `ν`-independent part of the inversion LHS: Coriolis, pressure
gradient, and divergence-free blocks. Pair with [`build_A_visc!`](@ref) to
add the viscous block; `A⁰ + A_visc` equals `build_A_inversion(fe_data,
params, eddy_param, b_vec)` for the `ν` implied by `b_vec`.

Since this never changes across a run (mesh, `f`, and the block structure are
fixed), it is assembled once via the ordinary `CellIterator` path — no need
for a cached flat kernel here, unlike the viscous block which is rebuilt
every few timesteps.
"""
function build_A_inversion_static(fe_data::FEData, params::Parameters)
    dh_up = fe_data.dh_up
    cv_u, cv_p, _ = make_cell_values(fe_data)
    n_u   = getnbasefunctions(cv_u)
    n_p   = getnbasefunctions(cv_p)
    n_loc = n_u + n_p
    f_cor = params.f

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
            f_q = f_cor(x)
            dΩ  = getdetJdV(cv_u, q)

            for i in 1:n_u
                φᵤ_i  = shape_value(cv_u, q, i)
                div_i = tr(shape_gradient(cv_u, q, i))

                for j in 1:n_u
                    φⱼ = shape_value(cv_u, q, j)
                    # Coriolis: f (ẑ×u)·v,  ẑ×u = (-u₂, u₁, 0)
                    cori = f_q * (-φⱼ[2] * φᵤ_i[1] + φⱼ[1] * φᵤ_i[2])
                    Ae[i, j] += cori * dΩ
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
    build_A_visc!(A, A⁰, fe_data, params, eddy_param, b_vec, nzidx_up)

Set `A.nzval = A⁰.nzval` then add the viscous block `∫ 2α²ε² ν ε(u):ε(v) dΩ`,
with `ν` from `ν_eddy` evaluated at `α(N² + ∂z(b))`. Runs on the flat,
`reinit!`-free [`AssemblyCache`](@ref) kernel: `ε(u):ε(v)` is written in closed
form from cached scalar-basis reference gradients, and results are scattered
directly into `A.nzval` via `nzidx_up` (the local u-u block → `A.nzval` index
map, from [`InversionLHSCache`](@ref)). `A`, `A⁰`, and `nzidx_up` must all
share the uncondensed inversion pattern (i.e. built from the same `fe_data`).
"""
function build_A_visc!(A::SparseMatrixCSC, A⁰::SparseMatrixCSC,
                       fe_data::FEData, params::Parameters,
                       eddy_param::EddyParameterization, b_vec::AbstractVector,
                       nzidx_up::Matrix{Int})
    cache = fe_data.cache
    nq   = length(cache.w)
    n_s  = size(cache.dphi_u, 2)   # scalar nodes per cell (10)
    n_u  = size(cache.dphi_u, 2) * 3
    n_b  = size(cache.phi_b, 2)
    ncells = length(cache.detJ)
    α²ε² = params.α^2 * params.ε^2
    α    = params.α
    N²   = params.N²

    A.nzval .= A⁰.nzval

    lb = zeros(n_b)
    g  = zeros(3, n_s)   # physical gradients of scalar basis fns at current quad point
    Ke = zeros(n_u, n_u)

    @inbounds for c in 1:ncells
        for i in 1:n_b
            lb[i] = b_vec[cache.dofs_b[i, c]]
        end
        Jᵀ = cache.Jinv_t[c]
        dJ = cache.detJ[c]
        x0 = cache.x0[c]
        J  = cache.J[c]
        fill!(Ke, 0.0)

        for q in 1:nq
            # ∂z(b) at this quad point
            gb1 = 0.0; gb2 = 0.0; gb3 = 0.0
            for i in 1:n_b
                gb1 += cache.dphi_b[1, i, q] * lb[i]
                gb2 += cache.dphi_b[2, i, q] * lb[i]
                gb3 += cache.dphi_b[3, i, q] * lb[i]
            end
            ∂z_b_q = _∂ᵣ(Jᵀ, 3, gb1, gb2, gb3)
            αbz_q  = α * (N² + ∂z_b_q)
            x_q    = x0 + J ⋅ cache.x_ref[q]   # affine map (linear tets)
            ν_q    = ν_eddy(eddy_param, eddy_param.f(x_q), αbz_q)
            νdΩ    = 2α²ε² * ν_q * cache.w[q] * dJ

            # physical gradients of the scalar P2 basis at this quad point
            for a in 1:n_s
                g[1, a] = _∂ᵣ(Jᵀ, 1, cache.dphi_u[1, a, q], cache.dphi_u[2, a, q], cache.dphi_u[3, a, q])
                g[2, a] = _∂ᵣ(Jᵀ, 2, cache.dphi_u[1, a, q], cache.dphi_u[2, a, q], cache.dphi_u[3, a, q])
                g[3, a] = _∂ᵣ(Jᵀ, 3, cache.dphi_u[1, a, q], cache.dphi_u[2, a, q], cache.dphi_u[3, a, q])
            end

            # ε(φ_i):ε(φ_j) = 1/2 [ δ_{cd}(g_a·g_b) + g_a[d] g_b[c] ],  i↔(a,c), j↔(b,d)
            for b in 1:n_s, a in 1:n_s
                dot_ab = g[1,a]*g[1,b] + g[2,a]*g[2,b] + g[3,a]*g[3,b]
                for d in 1:3
                    j = 3*(b - 1) + d
                    for c′ in 1:3
                        i = 3*(a - 1) + c′
                        econtract = 0.5 * ((c′ == d ? dot_ab : 0.0) + g[d, a] * g[c′, b])
                        Ke[i, j] += νdΩ * econtract
                    end
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

Returns a sparse matrix of size `(N_up × nb)` where N_up = ndofs(dh_up).
Only the u-rows are nonzero (pressure rows remain zero).
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
    A = build_A_inversion(fe_data, params, eddy_param, b_vec)

Assemble the inversion LHS with eddy viscosity computed at each quadrature point
from `∂z(b)` via `ν_eddy(eddy_param, α*(N² + ∂z(b)))`.
"""
function build_A_inversion(fe_data::FEData, params::Parameters,
                            eddy_param::EddyParameterization,
                            b_vec::AbstractVector)
    dh_up = fe_data.dh_up
    dh_b  = fe_data.dh_b
    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    n_u   = getnbasefunctions(cv_u)
    n_p   = getnbasefunctions(cv_p)
    n_loc = n_u + n_p
    α²ε²  = params.α^2 * params.ε^2
    f_cor = params.f
    α     = params.α
    N²    = params.N²

    A   = allocate_inversion_matrix(fe_data)
    asm = start_assemble(A)
    Ae  = zeros(n_loc, n_loc)

    for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(dh_b))
        reinit!(cv_u, cc_up)
        reinit!(cv_p, cc_up)
        reinit!(cv_b, cc_b)
        coords  = getcoordinates(cc_up)
        local_b = b_vec[celldofs(cc_b)]
        fill!(Ae, 0.0)

        for q in 1:getnquadpoints(cv_u)
            x       = spatial_coordinate(cv_u, q, coords)
            ∂z_b_q  = function_gradient(cv_b, q, local_b)[3]
            αbz_q   = α * (N² + ∂z_b_q)
            ν_q     = ν_eddy(eddy_param, eddy_param.f(x), αbz_q)
            f_q     = f_cor(x)
            dΩ      = getdetJdV(cv_u, q)

            for i in 1:n_u
                ε_i   = symmetric(shape_gradient(cv_u, q, i))
                φᵤ_i  = shape_value(cv_u, q, i)
                div_i = tr(shape_gradient(cv_u, q, i))
                for j in 1:n_u
                    φⱼ = shape_value(cv_u, q, j)
                    visc = 2α²ε² * ν_q * dcontract(ε_i, symmetric(shape_gradient(cv_u, q, j)))
                    cori = f_q * (-φⱼ[2] * φᵤ_i[1] + φⱼ[1] * φᵤ_i[2])
                    Ae[i, j] += (visc + cori) * dΩ
                end
                for j in 1:n_p
                    Ae[i, n_u + j] -= div_i * shape_value(cv_p, q, j) * dΩ
                end
            end
            for i in 1:n_p
                φ_p_i = shape_value(cv_p, q, i)
                for j in 1:n_u
                    Ae[n_u + i, j] += φ_p_i * tr(shape_gradient(cv_u, q, j)) * dΩ
                end
            end
        end
        assemble!(asm, celldofs(cc_up), Ae)
    end
    return A
end

"""
    f = build_f_wind(fe_data, params, forcings)

Assemble the wind-stress surface RHS:
`f[u_i] = ∫_Γ α (τˣ (x̂·φᵤ_i) + τʸ (ŷ·φᵤ_i)) dΓ`.
Returns a vector of length N_up; only u-rows on the surface facets are nonzero.
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
