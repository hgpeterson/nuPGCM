####
#### Preconditioners for the rotating-Stokes ("inversion") saddle-point system
####
####     𝒜 = [ K + N   -Bᵀ ]      K = ∫ 2η ε(u):ε(v),  η = α²ε²ν   (SPD)
####         [ B        0  ]      N = ∫ f (ẑ×u)·v                  (skew-symmetric)
####
#### Because `N` is exactly skew-symmetric, `𝒜` has positive-semidefinite
#### symmetric part -- the off-diagonal blocks cancel in `(𝒜 + 𝒜ᵀ)/2`, leaving
#### `diag(K, 0)`. Any SPD block-diagonal preconditioner therefore keeps the
#### preconditioned field of values in the right half-plane, which is what makes
#### GMRES well-behaved here *provided* the Schur block is right.
####
#### Getting the Schur block right is the whole game: see the header of
#### `pressure_operators.jl` for why the usual pressure mass matrix is wrong by a
#### factor `|k|²/k₃²` once rotation dominates viscosity, and what replaces it.
####
#### Everything here is applied as `y = P*x` via `mul!`, i.e. passed to Krylov.jl
#### as `M = P` with `ldiv = false`.
####

abstract type Preconditioner end

Base.eltype(::Preconditioner) = Float64

####
#### Approximate inverses: building blocks with `mul!(y, ai, x)` ≈ `y = A⁻¹x`
####

abstract type ApproxInverse end

Base.eltype(::ApproxInverse) = Float64

"""
    FactorInverse(fact)

Exact solve against a stored factorization (`lu`, `cholesky`, ...). The reference
point every approximate option is measured against; also the only option that
makes a "gold standard" direct comparison possible.
"""
struct FactorInverse{F} <: ApproxInverse
    fact::F
end
FactorInverse(A::SparseMatrixCSC; sym = false) =
    FactorInverse(sym ? cholesky(Symmetric(A)) : lu(A))

LinearAlgebra.mul!(y, ai::FactorInverse, x) = _fact_solve!(y, ai.fact, x)

# Most factorizations (UMFPACK `lu`, the GPU ILU(0)s) implement the in-place
# 3-argument `ldiv!`. CHOLMOD's sparse `Factor` does not, so it needs the
# allocating `\`. The allocation is one pressure-space vector per application
# (≈10 kB at np = 1234) — negligible next to the Krylov basis, and only on the
# pressure block.
_fact_solve!(y, F, x) = ldiv!(y, F, x)
_fact_solve!(y, F::SparseArrays.CHOLMOD.Factor, x) = (y .= F \ x; y)

"""
    DiagInverse(dinv)

Elementwise scaling `y = dinv .* x`. Used both as a Jacobi preconditioner and as
the lumped-mass `M⁻¹` inside composed pressure operators.
"""
struct DiagInverse{V} <: ApproxInverse
    dinv::V
end
DiagInverse(A::SparseMatrixCSC, arch::AbstractArchitecture) =
    DiagInverse(on_architecture(arch, 1 ./ diag(A)))
LinearAlgebra.mul!(y, ai::DiagInverse, x) = (y .= ai.dinv .* x; y)

"""
    ScaledIdentityInverse(c)

`y = c*x`. The current production preconditioner is `ScaledIdentityInverse(1/h³)`
applied to the whole system; kept so the benchmark can reproduce the baseline.
"""
struct ScaledIdentityInverse <: ApproxInverse
    c::Float64
end
LinearAlgebra.mul!(y, ai::ScaledIdentityInverse, x) = (y .= ai.c .* x; y)

"""
    FunctionInverse(f!, n)

Escape hatch wrapping any in-place `f!(y, x)`. Lets the benchmark scripts plug in
solvers that `nuPGCM` does not depend on (e.g. `AlgebraicMultigrid.jl`) without
adding them to the package.
"""
struct FunctionInverse{F} <: ApproxInverse
    f!::F
    n::Int
end
LinearAlgebra.mul!(y, ai::FunctionInverse, x) = ai.f!(y, x)

"""
    KrylovInverse(A, P; solver, itmax, ldiv, label, atol, rtol)

Fixed-budget inner Krylov solve, `y ≈ A⁻¹x` in at most `itmax` iterations
(`itmax = 0` means "run to tolerance"). `solver` is `:cg` for SPD `A` or
`:gmres` otherwise.

A truncated inner solve makes the *outer* preconditioner nonlinear, so the outer
Krylov method must be flexible. Use a fixed `itmax` and treat the result as a
fixed linear operator only when `itmax` is large enough that the inner solve is
effectively exact; otherwise prefer `restart = false` outer GMRES, or accept that
the outer iteration is really FGMRES-like.
"""
struct KrylovInverse{M, P, W} <: ApproxInverse
    A::M
    P::P
    workspace::W
    itmax::Int
    ldiv::Bool
    label::String
    atol::Float64
    rtol::Float64
    stats::Vector{Int}   # running tally of inner iterations, for benchmarking
end

function KrylovInverse(A, P; solver::Symbol = :cg, itmax::Int = 0, ldiv::Bool = false,
                       label::String = "", atol::Float64 = 1e-12, rtol::Float64 = 1e-8,
                       memory::Int = 20)
    arch = architecture(A)
    n    = size(A, 1)
    VT   = vector_type(arch, eltype(A))
    ws   = solver === :cg    ? Krylov.CgWorkspace(n, n, VT) :
           solver === :gmres ? Krylov.GmresWorkspace(n, n, VT; memory) :
           throw(ArgumentError("solver must be :cg or :gmres, got $solver"))
    ws.x .= zero(eltype(A))
    return KrylovInverse(A, P, ws, itmax, ldiv, label, atol, rtol, Int[])
end

function LinearAlgebra.mul!(y, ai::KrylovInverse, x)
    ws = ai.workspace
    fill!(ws.x, zero(eltype(ws.x)))
    Krylov.krylov_solve!(ws, ai.A, x, ws.x; M = ai.P, ldiv = ai.ldiv,
                         itmax = ai.itmax, atol = ai.atol, rtol = ai.rtol)
    push!(ai.stats, ws.stats.niter)
    y .= ws.x
    return y
end

"""
    HostInverse(inner, n, arch)

Runs a CPU `ApproxInverse` from inside a device-resident preconditioner, staging
the vector across the bus each way.

This is a deliberate design choice, not a fallback: the pressure space is ~5% of
the DOFs (≈8.3k of 185k at `h = 4e-2`), so a *direct* sparse factorization of a
pressure-space operator is cheap and exact, while the round trip moves only
`np` doubles (≈66 kB). Velocity-block work stays on the device.
"""
struct HostInverse{A <: ApproxInverse, V} <: ApproxInverse
    inner::A
    xh::Vector{Float64}
    yh::Vector{Float64}
    _dev::V   # type witness for the device vector; unused
end
function HostInverse(inner::ApproxInverse, n::Int, arch::AbstractArchitecture)
    return HostInverse(inner, zeros(n), zeros(n), on_architecture(arch, zeros(0)))
end
function LinearAlgebra.mul!(y, ai::HostInverse, x)
    copyto!(ai.xh, x)
    mul!(ai.yh, ai.inner, ai.xh)
    copyto!(y, ai.yh)
    return y
end

"""
    ComposedInverse(ops, buf)

Applies `ops` right-to-left: `y = ops[1] ∘ ops[2] ∘ … ∘ ops[end] x`. Each entry is
either an `ApproxInverse` (applied via `mul!`) or a matrix (applied as a matvec),
which is exactly what the composed Schur approximations below need — e.g.
`S̃⁻¹ = M⁻¹ K 𝒦⁻¹` is `ComposedInverse([M⁻¹, K, 𝒦⁻¹])`.
"""
struct ComposedInverse{T, V} <: ApproxInverse
    ops::T
    bufs::Vector{V}
end
function ComposedInverse(ops::Vector, n::Int, arch::AbstractArchitecture)
    bufs = [on_architecture(arch, zeros(n)) for _ in 1:2]
    return ComposedInverse(Tuple(ops), bufs)
end
function LinearAlgebra.mul!(y, ci::ComposedInverse, x)
    src = x
    dst = ci.bufs[1]
    alt = ci.bufs[2]
    n   = length(ci.ops)
    for i in n:-1:1
        out = (i == 1) ? y : dst
        mul!(out, ci.ops[i], src)
        if i > 1
            src, dst, alt = dst, alt, src
        end
    end
    return y
end

"""
    SumInverse(a, b, buf)

`y = a*x + b*x`. The Cahouet--Chabard construction is a sum of inverses: each
term is the inverse of one limiting behaviour of the Schur complement, and the
sum is spectrally equivalent to the inverse of the sum within a factor of 2 when
the terms commute.
"""
struct SumInverse{A, B, V} <: ApproxInverse
    a::A
    b::B
    buf::V
end
SumInverse(a, b, n::Int, arch::AbstractArchitecture) =
    SumInverse(a, b, on_architecture(arch, zeros(n)))
function LinearAlgebra.mul!(y, si::SumInverse, x)
    mul!(y, si.a, x)
    mul!(si.buf, si.b, x)
    y .+= si.buf
    return y
end

"""
    ScaledInverse(inner, c, ...)

`y = c * inner * x`.
"""
struct ScaledInverse{A} <: ApproxInverse
    inner::A
    c::Float64
end
function LinearAlgebra.mul!(y, si::ScaledInverse, x)
    mul!(y, si.inner, x)
    y .*= si.c
    return y
end

####
#### Whole-system preconditioners
####

"""
    ScaledIdentity(c, n)

`P = c·I`. Reproduces the current production preconditioner, `c = 1/h³`, whose
only job is to put the residual on a sensible scale. Included as the benchmark
baseline.
"""
struct ScaledIdentity <: Preconditioner
    c::Float64
    n::Int
end
LinearAlgebra.mul!(y, P::ScaledIdentity, x) = (y .= P.c .* x; y)
Base.size(P::ScaledIdentity) = (P.n, P.n)
Base.size(P::ScaledIdentity, ::Int) = P.n

"""
    RefreshablePreconditioner(inner, n)

Mutable indirection so a preconditioner can be swapped out mid-run without
touching `IterativeSolverToolkit`, whose `P` field is immutable by design (every
other container in it is updated in place).

Needed because the eddy parameterization makes `ν = ν(b)` drift, so any
preconditioner built from `ν` goes stale. `_update_eddy_A!` refreshes `solver.A`
every step but *not* `P` — harmless for the constant `Diagonal(1/h³)`, wrong for
anything ν-dependent. [`rebuild_preconditioner!`](@ref) assigns `inner` here.

The staleness measurement (`scratch/precond/staleness.jl`) says a Cahouet--Chabard
preconditioner frozen at step `i₀` costs +6% iterations after 50 steps, +15% after
100, and +84% after 150, while setup is ~1.3 s — so rebuilding every ~50 steps
amortizes to ~0.03 s/step and keeps the degradation in the noise.

The `mul!` here dispatches dynamically on `inner`; that is one indirection per
preconditioner application, negligible beside the solves it wraps.
"""
mutable struct RefreshablePreconditioner{P} <: Preconditioner
    inner::P
    n::Int
end
RefreshablePreconditioner(inner) = RefreshablePreconditioner{Any}(inner, size(inner, 1))
LinearAlgebra.mul!(y, P::RefreshablePreconditioner, x) = mul!(y, P.inner, x)
Base.size(P::RefreshablePreconditioner) = (P.n, P.n)
Base.size(P::RefreshablePreconditioner, ::Int) = P.n

"""
    BlockDiagonalPreconditioner(Ainv, Sinv, u_rng, p_rng; schur_sign)

    P = [ Ã   0  ]
        [ 0   σS̃ ]

with `Ã ≈ K+N` and `S̃ ≈ B A⁻¹ Bᵀ`. `schur_sign` (`σ = ±1`) is exposed because the
right sign depends on the saddle-point sign convention; with this code's
`[A -Bᵀ; B 0]` layout `+1` is correct, but `-1` is the classic choice for the
symmetric `[A Bᵀ; B 0]` form and is worth measuring.

Requires `FEData(...; dof_order = :blocked)` so that `u_rng` and `p_rng` are
contiguous — see [`block_ranges`](@ref).
"""
struct BlockDiagonalPreconditioner{AI, SI} <: Preconditioner
    Ainv::AI
    Sinv::SI
    u_rng::UnitRange{Int}
    p_rng::UnitRange{Int}
    schur_sign::Float64
end

function BlockDiagonalPreconditioner(Ainv, Sinv, u_rng, p_rng; schur_sign = 1.0)
    return BlockDiagonalPreconditioner(Ainv, Sinv, u_rng, p_rng, Float64(schur_sign))
end

function LinearAlgebra.mul!(y, P::BlockDiagonalPreconditioner, x)
    xu = @view x[P.u_rng]; yu = @view y[P.u_rng]
    xp = @view x[P.p_rng]; yp = @view y[P.p_rng]
    mul!(yu, P.Ainv, xu)
    mul!(yp, P.Sinv, xp)
    P.schur_sign == 1.0 || (yp .*= P.schur_sign)
    return y
end
Base.size(P::BlockDiagonalPreconditioner) = (last(P.p_rng), last(P.p_rng))
Base.size(P::BlockDiagonalPreconditioner, ::Int) = last(P.p_rng)

"""
    BlockTriangularPreconditioner(Ainv, Sinv, Aup, u_rng, p_rng, buf)

    P = [ Ã   A_up ]        A_up = the actual (1,2) block of 𝒜 (= -Bᵀ here)
        [ 0   S̃    ]

Applied as `y_p = S̃⁻¹x_p`, then `y_u = Ã⁻¹(x_u - A_up y_p)`.

With exact blocks, `𝒜P⁻¹ = [I 0; BA⁻¹  I]` is block-triangular with unit diagonal,
so GMRES terminates in two iterations. That is why the triangular variant beats
the diagonal one by roughly a factor of two in your Experiment Set 1 (15 → 7,
46 → 11, ...) and why it is the preferred outer shape here.
"""
struct BlockTriangularPreconditioner{AI, SI, M, V} <: Preconditioner
    Ainv::AI
    Sinv::SI
    Aup::M
    u_rng::UnitRange{Int}
    p_rng::UnitRange{Int}
    buf::V
end

# `arch` is a keyword, not a 6th positional argument: as a positional it would be
# ambiguous with the struct's own default constructor whenever the buffer type
# parameter `V` could itself be an architecture.
function BlockTriangularPreconditioner(Ainv, Sinv, Aup, u_rng, p_rng;
                                       arch::AbstractArchitecture = CPU())
    return BlockTriangularPreconditioner(Ainv, Sinv, Aup, u_rng, p_rng,
                                         on_architecture(arch, zeros(length(u_rng))))
end

function LinearAlgebra.mul!(y, P::BlockTriangularPreconditioner, x)
    xu = @view x[P.u_rng]; yu = @view y[P.u_rng]
    xp = @view x[P.p_rng]; yp = @view y[P.p_rng]
    mul!(yp, P.Sinv, xp)
    # buf = x_u - A_up * y_p
    P.buf .= xu
    mul!(P.buf, P.Aup, yp, -1.0, 1.0)
    mul!(yu, P.Ainv, P.buf)
    return y
end
Base.size(P::BlockTriangularPreconditioner) = (last(P.p_rng), last(P.p_rng))
Base.size(P::BlockTriangularPreconditioner, ::Int) = last(P.p_rng)

"""
    NullspaceProjected(inner, p_rng, w)

Wraps a preconditioner so that every application projects the constant-pressure
mode out of the result:

    y = P⁻¹x ,   y_p ← y_p - w (wᵀy_p)/(wᵀw)

For `pressure_gauge = :none` the inversion matrix is singular with
`N(𝒜) = N(𝒜ᵀ) = span{(0, 1_p)}` and a consistent RHS, so Krylov methods converge
provided iterates stay in the orthogonal complement — which this enforces.

Preferred over `:pin`: pinning kills the exact null mode but leaves a near-null
mode ("constant except at the pinned DOF") with a spuriously tiny Schur
eigenvalue that a direct solve never notices but GMRES pays for.
"""
struct NullspaceProjected{P <: Preconditioner, V} <: Preconditioner
    inner::P
    p_rng::UnitRange{Int}
    w::V
    wnorm2::Float64
end

function NullspaceProjected(inner::Preconditioner, p_rng::UnitRange{Int},
                            arch::AbstractArchitecture; weights = nothing)
    w = weights === nothing ? ones(length(p_rng)) : Vector{Float64}(weights)
    return NullspaceProjected(inner, p_rng, on_architecture(arch, w), sum(abs2, w))
end

function LinearAlgebra.mul!(y, P::NullspaceProjected, x)
    mul!(y, P.inner, x)
    yp = @view y[P.p_rng]
    c  = dot(P.w, yp) / P.wnorm2
    yp .-= c .* P.w
    return y
end
Base.size(P::NullspaceProjected) = size(P.inner)
Base.size(P::NullspaceProjected, i::Int) = size(P.inner, i)

"""
    ProjectedInverse(inner, w, buf)

Pseudo-inverse: projects `w` out of both the input and the output,
`y = P⊥ inner P⊥ x` with `P⊥ = I - wwᵀ/wᵀw`.

Essential for every pressure-space operator that is singular on the constant mode
(`Kf`, `𝒦`, `BTBᵀ`). Without it, `X⁻¹` amplifies that mode by `1/λ_min` — which is
`~1e12` under the `:pin` gauge — and the amplified component either swamps the
result outright or, in a composition like `K 𝒦⁻¹` where `K` annihilates the same
mode, cancels 12 digits and leaves noise. Both were observed: `stiffness` and
`geostrophic` stalled at relative residual ≈ 1 until this projection was added,
while `cahouet_chabard` survived only because its well-scaled `Mν⁻¹` term
dominated the sum.

Projecting is also the physically right thing: `S` itself annihilates the constant
pressure, so `S̃⁻¹` should put no energy there.
"""
struct ProjectedInverse{A <: ApproxInverse, V} <: ApproxInverse
    inner::A
    w::V
    buf::V
end
function ProjectedInverse(inner::ApproxInverse, n::Int)
    w = ones(n) ./ sqrt(n)
    return ProjectedInverse(inner, w, zeros(n))
end
function LinearAlgebra.mul!(y, pi::ProjectedInverse, x)
    pi.buf .= x .- pi.w .* dot(pi.w, x)
    mul!(y, pi.inner, pi.buf)
    y .-= pi.w .* dot(pi.w, y)
    return y
end

"""
    _spd_inverse(X, M; σ, label)

Factorize a pressure-space operator that is singular, or nearly so, on the
constant-pressure mode.

`Kz + δKh`, `Kf = ∫(1/|f|)∇p·∇q` and `L = BTBᵀ` all annihilate a constant
pressure: the first two because a constant has no gradient, the last because
`Bᵀ1 = 0` whenever every admissible velocity satisfies `∮v·n = 0` (true here —
`u = 0` on the bottom and sidewalls, `w = 0` at the surface, and the periodic
faces cancel). So:

- under `pressure_gauge = :none` these are *exactly* singular and Cholesky throws;
- under `:pin` they are merely near-singular, because the reduced space excludes
  the pinned DOF and "constant on the free DOFs" is no longer a true constant.

Adding `σ M` (relative to the operator's own diagonal scale) shifts that mode off
zero without perturbing anything else — `σ M` is a zeroth-order term, so it is
negligible wherever the operator has any gradient content at all. The mode itself
is handled properly by [`NullspaceProjected`](@ref) when the `:none` gauge is used.

Falls back to `lu` if the shifted operator still fails Cholesky.
"""
function _spd_inverse(X::SparseMatrixCSC, M::SparseMatrixCSC; σ::Float64 = 1e-8,
                      label::String = "", project::Bool = false)
    n = size(X, 1)
    if σ > 0
        dX = diag(X); dM = diag(M)
        sX = sum(dX) / max(count(!iszero, dX), 1)
        sM = sum(dM) / max(count(!iszero, dM), 1)
        X = X + (σ * sX / sM) * M
    end
    ai = try
        FactorInverse(cholesky(Symmetric(X)))
    catch
        @warn "Cholesky failed for pressure operator $label; falling back to lu" maxlog=3
        FactorInverse(lu(X))
    end
    return project ? ProjectedInverse(ai, n) : ai
end

####
#### Schur-complement approximations
####
#### Each returns an `ApproxInverse` acting on the reduced pressure space with
#### `mul!(y, Sinv, x)` ≈ `y = S̃⁻¹x`. `S = B A⁻¹ Bᵀ` is positive definite in this
#### sign convention.
####

"""
    schur_mass(po, arch; exact)

    S̃ = ∫ p q / (2η)   ⇒   S̃⁻¹ = Mν⁻¹

The textbook Stokes choice (viscosity-weighted pressure mass matrix). Correct in
the viscous limit `η|k|² ≫ f`, and **wrong by `|k|²/k₃²` once rotation dominates**
— it is included to reproduce and quantify the failure, not because it should be
used. `exact = false` uses the lumped diagonal, which is spectrally equivalent for
P1 and much cheaper.
"""
function schur_mass(po::PressureOperators, arch::AbstractArchitecture; exact::Bool = false)
    if exact
        return HostInverse(FactorInverse(po.Mν; sym = true), size(po.Mν, 1), arch)
    end
    return DiagInverse(on_architecture(arch, 1 ./ vec(sum(po.Mν, dims = 2))))
end

"""
    schur_stiffness(po, arch)

    S̃⁻¹ = Kf⁻¹ ,   Kf = ∫ (1/|f|) ∇p·∇q

The `1/f`-weighted pressure stiffness matrix. Carries the `1/|k|²` factor the mass
matrix lacks, which is why it survived where the mass matrix failed in Experiment
Set 1 (PG Thin BL: 1,819 iterations, no breakdown). Cheap: one Poisson solve.
"""
function schur_stiffness(po::PressureOperators, arch::AbstractArchitecture;
                         inner = nothing, σ::Float64 = 1e-8)
    n = size(po.Kf, 1)
    ai = inner === nothing ? _spd_inverse(po.Kf, po.M; σ, label = "Kf") : inner
    return HostInverse(ai, n, arch)
end

"""
    schur_cahouet_chabard(po, arch)

    S̃⁻¹ = Mν⁻¹ + Kf⁻¹

Cahouet--Chabard, transplanted to rotation by treating the Coriolis term as if it
were a symmetric zeroth-order term of size `|f|`. The two terms are the viscous
and rotating limits of `S⁻¹`; summing inverses interpolates between them.

This is *not* the correct rotating-limit operator (the true one is anisotropic,
see [`schur_geostrophic`](@ref)) but it is the cheapest thing that is not simply
wrong, and it is the natural generalization of your Experiment Set 1 result.
"""
function schur_cahouet_chabard(po::PressureOperators, arch::AbstractArchitecture;
                               inner_K = nothing, σ::Float64 = 1e-8)
    n  = size(po.M, 1)
    Mi = DiagInverse(on_architecture(arch, 1 ./ vec(sum(po.Mν, dims = 2))))
    Ki = schur_stiffness(po, arch; inner = inner_K, σ)
    return SumInverse(Mi, Ki, n, arch)
end

"""
    schur_geostrophic(po, arch; δ, full, inner)

The rotation-aware Schur approximation derived in `pressure_operators.jl`:

    S̃⁻¹ = M⁻¹ K 𝒦⁻¹

Two choices of `𝒦`:

- `full = false` (default) — **anisotropic Poisson**

      𝒦 = Kz_ην + δ·Kh_ην       symbol:  (k₃² + δ k_h²)/η

  so `S̃⁻¹` has symbol `η|k|²/(k₃² + δ k_h²)`, i.e. exactly the missing `|k|²/k₃²`
  factor, regularized by `δ` so the barotropic modes (`k₃ = 0`) stay bounded.
  One anisotropic Poisson solve per application; AMG-friendly; `δ` is the single
  tuning knob.

  **Choosing `δ`.** Matching the horizontal part of `S̃` to the exact symbol,
  `δ k_h²/(η|k|²) = a k_h²/(a² + f²)` with `a = η|k|²`, gives

      δ(k) = a² / (a² + f²)  ∈ (0, 1]

  so `δ → 1` in the viscous limit and `δ → (a/f)²` in the rotating limit. Two
  consequences that are easy to get wrong:

  * `δ = 1` makes `𝒦 = K` and `S̃ = M/η` — the plain **mass matrix**. The
    geostrophic family therefore *contains* the mass matrix and can only improve
    on it if `δ` is tuned; it is not a different animal.
  * At the grid scale `a > f` here, so `δ ≈ 1` there, while at domain scale
    `δ ~ 1e-4`. A single constant `δ` is a compromise, and picking it near the
    *domain*-scale value over-amplifies grid-scale barotropic modes by `~1/δ` and
    stalls GMRES outright. Useful constant values live in `[1e-3, 1]`, not below.

  For a `δ` that is correct at every scale, use `full = true`.

- `full = true` — the untruncated symbol

      𝒦 = Kz_ην + Kh_ν_f2 · M⁻¹K · M⁻¹K       symbol:  k₃²/η + (η k_h²/f²)|k|⁴

  Matches `s₀` with no free parameter, at the cost of a 6th-order operator whose
  sparsity is much denser and which no multigrid will love.
"""
function schur_geostrophic(po::PressureOperators, arch::AbstractArchitecture;
                           δ::Real = 1e-3, full::Bool = false, inner = nothing,
                           σ::Float64 = 1e-8)
    n     = size(po.M, 1)
    Minv  = Diagonal(1 ./ po.Mlump)
    𝒦 = if full
        MK = Minv * po.K
        po.Kz_ην + po.Kh_ν_f2 * MK * MK
    else
        po.Kz_ην + δ * po.Kh_ην
    end
    𝒦inv = inner === nothing ? _spd_inverse(sparse(𝒦), po.M; σ, label = "K_geo") : inner
    # S̃⁻¹ = M⁻¹ K 𝒦⁻¹, all on the host (np is small)
    host = ComposedInverse(Any[DiagInverse(1 ./ po.Mlump), po.K, 𝒦inv], n, CPU())
    return HostInverse(host, n, arch)
end

"""
    schur_lsc(po, B, A_uu, Mu_lump, arch)

Least-squares commutator (BFBt/LSC):

    S̃⁻¹ = (B T Bᵀ)⁻¹ (B T A T Bᵀ) (B T Bᵀ)⁻¹ ,   T = diag(M_u)⁻¹

Makes no commutator assumption about the structure of `A`, so unlike PCD it is not
invalidated by the skew Coriolis block. In Experiment Set 1 this was the only
approximation that reached the true solution in the PG Thin BL case (~2,000
iterations, no breakdown).

`B T Bᵀ` is a `np × np` pressure-Laplacian-like SPD matrix, formed and factorized
once on the host.
"""
function schur_lsc(B::SparseMatrixCSC, A_uu::SparseMatrixCSC, Mu_lump::Vector{Float64},
                   Mp::SparseMatrixCSC, arch::AbstractArchitecture;
                   inner = nothing, σ::Float64 = 1e-8)
    n  = size(B, 1)
    T  = Diagonal(1 ./ Mu_lump)
    BT = B * T
    L  = sparse(BT * B')                 # B T Bᵀ — singular on the constant mode
    Linv = inner === nothing ? _spd_inverse(L, Mp; σ, label = "BTBt") : inner
    Mid  = sparse(BT * A_uu * T * B')    # B T A T Bᵀ
    host = ComposedInverse(Any[Linv, Mid, Linv], n, CPU())
    return HostInverse(host, n, arch)
end

"""
    schur_augmented_lagrangian(po, γ, arch; base)

    S_γ⁻¹ = S₀⁻¹ + γ W⁻¹     (exact, Sherman--Morrison--Woodbury)

For the augmented system `A_γ = A + γBᵀW⁻¹B` with `W = M_p`, this identity is
*exact*, so `S_γ⁻¹ ≈ γM_p⁻¹` becomes accurate for large `γ` **regardless of what
sits in the (1,1) block** — including Coriolis. That parameter-robustness is the
reason AL is the literature's answer to (1,1)-blocks that break commutator
arguments.

`base` supplies the `S₀⁻¹` term; passing a rotation-aware approximation (rather
than the usual `2ηM_p⁻¹` Stokes limit) lets `γ` stay much smaller than
`f²/(ηk_h²)`, which is what keeps the augmented velocity solve tractable.
"""
function schur_augmented_lagrangian(po::PressureOperators, γ::Real,
                                    arch::AbstractArchitecture; base = nothing)
    n  = size(po.M, 1)
    Mi = ScaledInverse(DiagInverse(on_architecture(arch, 1 ./ po.Mlump)), Float64(γ))
    base === nothing && return Mi
    return SumInverse(base, Mi, n, arch)
end

"""
    schur_exact(A_uu_inv, B, arch, np)

`S̃⁻¹` = exact inverse of `S = B A⁻¹ Bᵀ`, formed densely. Requires `np` solves
against `A`, so it is only usable on the smallest meshes — it exists to validate
the analysis (a block-triangular preconditioner with the exact Schur block must
converge in ≤ 2 GMRES iterations) and to measure how far each approximation is
from optimal.
"""
function schur_exact(A_uu::SparseMatrixCSC, B::SparseMatrixCSC, arch::AbstractArchitecture)
    np   = size(B, 1)
    fact = lu(A_uu)
    Bt   = Matrix(B')
    X    = fact \ Bt              # A⁻¹Bᵀ, nu × np
    S    = Matrix(B * X)          # B A⁻¹ Bᵀ
    return HostInverse(FactorInverse(lu(S)), np, arch)
end

####
#### Block extraction and augmented-Lagrangian reformulation
####

"""
    InversionBlocks

The 2×2 blocks of the reduced inversion matrix, on the host. `A_up == -A_pu'`
exactly (the discretization emits `-∫(∇·v)p` and `+∫q(∇·u)` from the same
quadrature loop), so `B = A_pu` and `Bᵀ = A_pu'`.
"""
struct InversionBlocks
    A_uu::SparseMatrixCSC{Float64, Int}
    A_up::SparseMatrixCSC{Float64, Int}
    A_pu::SparseMatrixCSC{Float64, Int}
    u_rng::UnitRange{Int}
    p_rng::UnitRange{Int}
end

"""
    blocks = split_blocks(A_red, fe_data)

Slice the reduced inversion matrix into velocity/pressure blocks. Requires
`dof_order = :blocked`.
"""
function split_blocks(A_red::SparseMatrixCSC, fe_data::FEData)
    u_rng, p_rng = block_ranges(fe_data)
    return InversionBlocks(A_red[u_rng, u_rng], A_red[u_rng, p_rng],
                           A_red[p_rng, u_rng], u_rng, p_rng)
end

"""
    A_γ = augment_system(A_red, blocks, W_lump, γ)

Augmented-Lagrangian reformulation: add `γ Bᵀ W⁻¹ B` to the (1,1) block, with `W`
the lumped pressure mass matrix.

**The solution is unchanged.** The augmented momentum equation is
`A u + Bᵀp + γBᵀW⁻¹(Bu) = f + γBᵀW⁻¹g`, and here the pressure-row RHS is
identically zero (`build_B_inversion` fills only u-rows, `f_wind` is a surface
velocity integral, and `f_bc` vanishes because every velocity Dirichlet value is
zero), so `g = 0` and the RHS needs no correction at all. The benchmark asserts
this against a direct solve rather than taking it on trust.
"""
function augment_system(A_red::SparseMatrixCSC, blocks::InversionBlocks,
                        W_lump::Vector{Float64}, γ::Real)
    γ == 0 && return A_red
    B   = blocks.A_pu
    aug = γ * (B' * Diagonal(1 ./ W_lump) * B)     # nu × nu
    # Scatter into a full-size sparse matrix and add. Doing this as
    # `A_γ[u_rng, u_rng] += aug` instead would go through SparseMatrixCSC's
    # sub-block `setindex!`, which rebuilds the pattern per entry and is orders
    # of magnitude slower.
    n   = size(A_red, 1)
    off = first(blocks.u_rng) - 1
    I_, J_, V_ = findnz(aug)
    return A_red + sparse(I_ .+ off, J_ .+ off, V_, n, n)
end

####
#### Velocity-block approximate inverses
####

"""
    velocity_inverse(kind, arch, A_uu_cpu, A_uu_dev; kwargs...)

Build `Ã⁻¹ ≈ (K+N)⁻¹`. `kind` is one of

| kind          | meaning                                                       |
|---------------|---------------------------------------------------------------|
| `:lu`         | exact sparse LU (host; wrapped in `HostInverse` on GPU)        |
| `:ilu0`       | ILU(0) — `kp_ilu0` on device, `ilu(A; τ=0)`-like on host       |
| `:ilut`       | threshold ILU on the host, drop tolerance `τ`                  |
| `:diag`       | Jacobi                                                         |
| `:scaled_id`  | `c·I` (the production `1/h³` baseline)                         |
| `:krylov`     | inner GMRES, `itmax` iterations, preconditioned by `inner_kind`|

Note `A_uu` is nonsymmetric (skew Coriolis block), so the inner solver is GMRES,
never CG.
"""
function velocity_inverse(kind::Symbol, arch::AbstractArchitecture,
                          A_uu_cpu::SparseMatrixCSC, A_uu_dev;
                          τ::Float64 = 1e-3, c::Float64 = 1.0, itmax::Int = 5,
                          inner_kind::Symbol = :ilu0, memory::Int = 20)
    n = size(A_uu_cpu, 1)
    if kind === :lu
        return HostInverse(FactorInverse(lu(A_uu_cpu)), n, arch)
    elseif kind === :ilu0
        if arch isa GPU
            return FactorInverse(KrylovPreconditioners.kp_ilu0(A_uu_dev))
        else
            # There is no true ILU(0) on the host here: KrylovPreconditioners' `ilu`
            # is a Crout ILU with a *drop tolerance*, and τ = 0 drops nothing, i.e.
            # it degenerates into a complete LU with full fill-in. Fall back to a
            # threshold ILU instead of silently building something ruinous.
            @warn "no host ILU(0) available; using threshold ILU with τ = $τ (use :ilut to set τ)" maxlog=1
            return HostInverse(FactorInverse(KrylovPreconditioners.ilu(A_uu_cpu; τ)), n, arch)
        end
    elseif kind === :ilut
        return HostInverse(FactorInverse(KrylovPreconditioners.ilu(A_uu_cpu; τ = τ)), n, arch)
    elseif kind === :diag
        return DiagInverse(on_architecture(arch, 1 ./ diag(A_uu_cpu)))
    elseif kind === :scaled_id
        return ScaledIdentityInverse(c)
    elseif kind === :krylov
        inner = velocity_inverse(inner_kind, arch, A_uu_cpu, A_uu_dev; τ, c)
        ldiv  = inner isa FactorInverse
        return KrylovInverse(A_uu_dev, ldiv ? inner.fact : inner;
                             solver = :gmres, itmax, ldiv, label = "A-block", memory)
    else
        throw(ArgumentError("unknown velocity-block kind :$kind"))
    end
end

####
#### Factory
####

"""
    P, meta = build_preconditioner(spec, arch, fe_data, params, forcings,
                                   A_red_cpu, A_red_dev; b_vec, h)

Assemble a named preconditioner from a `NamedTuple` spec. Returns the
preconditioner and a `meta` `NamedTuple` of setup timings and sizes for the
benchmark table.

Recognised `spec` fields:

- `kind`      — `:none`, `:scaled_id`, `:block_diag`, `:block_tri`
- `avel`      — velocity block, see [`velocity_inverse`](@ref)
- `schur`     — `:mass`, `:stiffness`, `:cahouet_chabard`, `:geostrophic`,
                `:geostrophic_full`, `:lsc`, `:al`, `:exact`
- `δ`         — regularization for `:geostrophic`
- `γ`         — augmentation parameter for `:al` (system must be augmented too)
- `itmax`, `τ`, `inner_kind`, `memory` — passed through to the velocity block
- `project`   — project the constant-pressure mode out of every application
- `schur_sign`— sign of the Schur block in `:block_diag`

`h` is the median edge length, used only by the `:scaled_id` baseline.
"""
function build_preconditioner(spec::NamedTuple, arch::AbstractArchitecture,
                              fe_data::FEData, params::Parameters, forcings::Forcings,
                              A_red_cpu::SparseMatrixCSC, A_red_dev;
                              b_vec = nothing, h::Float64 = 1.0,
                              po::Union{Nothing, PressureOperators} = nothing,
                              Mu_lump::Union{Nothing, Vector{Float64}} = nothing)
    kind = get(spec, :kind, :block_tri)
    n    = size(A_red_cpu, 1)

    if kind === :none
        return ScaledIdentity(1.0, n), (; setup_total = 0.0)
    elseif kind === :scaled_id
        return ScaledIdentity(1 / h^3, n), (; setup_total = 0.0)
    end

    t0 = time()
    blocks = split_blocks(A_red_cpu, fe_data)
    u_rng, p_rng = blocks.u_rng, blocks.p_rng
    t_split = time() - t0

    t0 = time()
    po === nothing && (po = build_pressure_operators(fe_data, params, forcings; b_vec))
    t_press = time() - t0

    # --- Schur block -------------------------------------------------------
    t0 = time()
    sk = get(spec, :schur, :geostrophic)
    Sinv = if sk === :mass
        schur_mass(po, arch; exact = get(spec, :exact_mass, false))
    elseif sk === :stiffness
        schur_stiffness(po, arch)
    elseif sk === :cahouet_chabard
        schur_cahouet_chabard(po, arch)
    elseif sk === :geostrophic
        schur_geostrophic(po, arch; δ = get(spec, :δ, 1e-3), full = false)
    elseif sk === :geostrophic_full
        schur_geostrophic(po, arch; full = true)
    elseif sk === :lsc
        Mu_lump === nothing && (Mu_lump = build_velocity_mass_lumped(fe_data))
        schur_lsc(blocks.A_pu, blocks.A_uu, Mu_lump, po.M, arch)
    elseif sk === :al
        base = get(spec, :al_base, :mass) === :none ? nothing :
               (get(spec, :al_base, :mass) === :geostrophic ?
                    schur_geostrophic(po, arch; δ = get(spec, :δ, 1e-3)) :
                    schur_mass(po, arch))
        schur_augmented_lagrangian(po, get(spec, :γ, 1.0), arch; base)
    elseif sk === :exact
        schur_exact(blocks.A_uu, blocks.A_pu, arch)
    else
        throw(ArgumentError("unknown schur kind :$sk"))
    end
    t_schur = time() - t0

    # --- velocity block ----------------------------------------------------
    t0 = time()
    A_uu_dev = arch isa GPU ? on_architecture(arch, blocks.A_uu) : blocks.A_uu
    Ainv = velocity_inverse(get(spec, :avel, :ilu0), arch, blocks.A_uu, A_uu_dev;
                            τ          = get(spec, :τ, 1e-3),
                            c          = 1 / h^3,
                            itmax      = get(spec, :itmax, 5),
                            inner_kind = get(spec, :inner_kind, :ilu0),
                            memory     = get(spec, :memory, 20))
    t_vel = time() - t0

    # --- assemble outer block preconditioner -------------------------------
    P = if kind === :block_diag
        BlockDiagonalPreconditioner(Ainv, Sinv, u_rng, p_rng;
                                    schur_sign = get(spec, :schur_sign, 1.0))
    elseif kind === :block_tri
        A_up_dev = arch isa GPU ? on_architecture(arch, blocks.A_up) : blocks.A_up
        BlockTriangularPreconditioner(Ainv, Sinv, A_up_dev, u_rng, p_rng; arch)
    else
        throw(ArgumentError("unknown preconditioner kind :$kind"))
    end

    if get(spec, :project, false)
        P = NullspaceProjected(P, p_rng, arch)
    end

    meta = (; t_split, t_press, t_schur, t_vel,
              setup_total = t_split + t_press + t_schur + t_vel,
              nu = length(u_rng), np = length(p_rng))
    return P, meta
end
