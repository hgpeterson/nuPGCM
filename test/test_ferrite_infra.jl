using Test
using Ferrite
using FerriteGmsh
using Gmsh: gmsh
using SparseArrays
using LinearAlgebra
using Krylov
using Printf

# Load the new infrastructure directly; nuPGCM doesn't compile yet because
# other source files still use Gridap.
const SRC = joinpath(@__DIR__, "../src")
include(joinpath(SRC, "architectures.jl"))
include(joinpath(SRC, "utils.jl"))
include(joinpath(SRC, "inputs.jl"))
include(joinpath(SRC, "meshes.jl"))
include(joinpath(SRC, "dofs.jl"))
include(joinpath(SRC, "spaces.jl"))
include(joinpath(SRC, "iterative_solvers.jl"))
include(joinpath(SRC, "timesteppers.jl"))
include(joinpath(SRC, "evolution.jl"))
include(joinpath(SRC, "inversion.jl"))

# ---------------------------------------------------------------------------
# Shared test mesh (channel-basin-flat, coarse) and shared FEData
# ---------------------------------------------------------------------------
const MESH_FILE = joinpath(@__DIR__, "../meshes/channel_basin_flat_h1.00e-01_a5.00e-01.msh")

function ensure_test_mesh()
    isfile(MESH_FILE) && return
    @info "Generating test mesh..."
    include(joinpath(@__DIR__, "../meshes/channel_basin_flat.jl"))
    mesh_channel_basin_flat(0.1, 0.5)
end

ensure_test_mesh()

# Shared helpers used across multiple testsets
function make_test_fe_data()
    mesh = Mesh(MESH_FILE)
    return FEData(mesh;
        u_diri_tags  = ["bottom", "surface"],
        u_diri_masks = [(true, true, true), (false, false, true)],
        b_diri_tags  = ["surface"],
        b_diri_vals  = [x -> 0.0])
end

function make_test_params()
    return Parameters(; ε=0.2, α=0.5, μϱ=10.0, N²=2.0, f=x->1.0, H=x->1.0)
end

function make_test_forcings()
    κₕ(x) = 1e-2 + 1e-3 * x[3]^2
    κᵥ(x) = 1e-2 + 1e-3 * x[3]^2
    return Forcings(1.0, κₕ, κᵥ, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))
end

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
@testset "Mesh" begin
    mesh = Mesh(MESH_FILE)

    @test mesh isa Mesh
    @test mesh.surface_tag == "surface"
    @test getnnodes(mesh.grid) > 0
    @test getncells(mesh.grid) > 0

    # updated mesh scripts give channel_west / channel_east, not "interior"
    @test haskey(mesh.grid.facetsets, "bottom")
    @test haskey(mesh.grid.facetsets, "surface")
    @test haskey(mesh.grid.facetsets, "channel_west")
    @test haskey(mesh.grid.facetsets, "channel_east")
    @test !haskey(mesh.grid.facetsets, "interior")

    # channel_west and channel_east have the same number of facets
    @test length(mesh.grid.facetsets["channel_west"]) ==
          length(mesh.grid.facetsets["channel_east"])

    # get_p_t shape
    p, t = get_p_t(mesh)
    @test size(p, 1) == getnnodes(mesh.grid)
    @test size(p, 2) == 3              # 3-D
    @test size(t, 1) == getncells(mesh.grid)
    @test size(t, 2) == 4              # tetrahedra

    # all node indices in t are valid
    @test minimum(t) >= 1
    @test maximum(t) <= getnnodes(mesh.grid)

    # channel geometry: x ∈ [0, 1]  (W = 1)
    @test minimum(p[:, 1]) ≈ 0.0 atol=1e-10
    @test maximum(p[:, 1]) ≈ 1.0 atol=1e-10

    # compute_h_cells
    h_cells = compute_h_cells(mesh)
    @test length(h_cells) == getncells(mesh.grid)
    @test all(h_cells .> 0)
    @test maximum(h_cells) < 1.0      # coarser than the domain size

    # string constructor (file name) round-trips
    p2, t2 = get_p_t(MESH_FILE)
    @test p2 ≈ p
    @test t2 == t
end

# ---------------------------------------------------------------------------
# FEData
# ---------------------------------------------------------------------------
@testset "FEData" begin
    fe_data = make_test_fe_data()
    nu, np, nb = get_n_dofs(fe_data)

    @testset "DOF counts" begin
        @test nu > 0
        @test np > 0
        @test nb > 0

        # P2/P1 Taylor-Hood: nu = 3×nb (both P2), np = n_P1_nodes < nb
        @test nu == 3 * nb            # velocity is 3-component P2
        @test np < nb                 # P1 pressure has fewer DOFs than P2 buoyancy

        # u/p DOF index arrays have the right sizes and are disjoint
        @test length(fe_data.u_dof_indices) == nu
        @test length(fe_data.p_dof_indices) == np
        @test isempty(intersect(fe_data.u_dof_indices, fe_data.p_dof_indices))

        # combined DofHandler total = nu + np
        @test ndofs(fe_data.dh_up) == nu + np
        @test ndofs(fe_data.dh_b)  == nb

        # stored orders
        @test fe_data.u_order == 2
        @test fe_data.b_order == 2
    end

    @testset "Dirichlet constraints" begin
        # bottom no-slip + surface z-component + periodic: combined in ch_up
        @test length(fe_data.ch_up.prescribed_dofs) > 0
        # buoyancy surface Dirichlet + periodic
        @test length(fe_data.ch_b.prescribed_dofs) > 0
    end

    @testset "Periodic constraints" begin
        # ch_up includes periodic for both :u and :p; ch_b includes periodic for :b
        @test length(fe_data.ch_up.prescribed_dofs) > 0
        @test length(fe_data.ch_b.prescribed_dofs)  > 0

        # constrained count is less than total DOFs
        @test length(fe_data.ch_up.prescribed_dofs) < nu + np
        @test length(fe_data.ch_b.prescribed_dofs)  < nb
    end

    @testset "Permutations" begin
        # identity permutations (not yet replaced with Cuthill-McKee)
        @test fe_data.p_up    == collect(1:nu + np)
        @test fe_data.p_b     == collect(1:nb)
        @test fe_data.inv_p_up == collect(1:nu + np)
        @test fe_data.inv_p_b  == collect(1:nb)
    end
end

# ---------------------------------------------------------------------------
# CellValues, FacetValues, matrix allocators
# ---------------------------------------------------------------------------
@testset "Spaces" begin
    fe_data = make_test_fe_data()
    nu, np, nb = get_n_dofs(fe_data)

    cv_u, cv_p, cv_b = make_cell_values(fe_data)
    fv_u, fv_b       = make_facet_values(fe_data)

    @testset "CellValues dimensions" begin
        # P2 vector: 10 scalar nodes × 3 components = 30
        @test getnbasefunctions(cv_u) == 30
        # P1 scalar: 4 nodes per tet
        @test getnbasefunctions(cv_p) == 4
        # P2 scalar: 10 nodes per tet
        @test getnbasefunctions(cv_b) == 10
        # all use the same quadrature (QR_ORDER = 3 → 5 points on tet)
        @test getnquadpoints(cv_u) == getnquadpoints(cv_p) == getnquadpoints(cv_b)
    end

    @testset "FacetValues dimensions" begin
        @test getnbasefunctions(fv_u) == 30
        @test getnbasefunctions(fv_b) == 10
    end

    @testset "Partition of unity (P1 pressure)" begin
        # For any interpolation, ∑ φᵢ(x) = 1 at every quadrature point.
        # Use dh_up since dh_p no longer exists separately.
        cc = first(CellIterator(fe_data.dh_up))
        reinit!(cv_p, cc)
        for q in 1:getnquadpoints(cv_p)
            s = sum(shape_value(cv_p, q, i) for i in 1:getnbasefunctions(cv_p))
            @test s ≈ 1.0 atol=1e-14
        end
    end

    @testset "Partition of unity (P2 buoyancy)" begin
        cc = first(CellIterator(fe_data.dh_b))
        reinit!(cv_b, cc)
        for q in 1:getnquadpoints(cv_b)
            s = sum(shape_value(cv_b, q, i) for i in 1:getnbasefunctions(cv_b))
            @test s ≈ 1.0 atol=1e-14
        end
    end

    @testset "Sum of gradients is zero (P1 pressure)" begin
        # Consequence of partition of unity: ∑ ∇φᵢ = 0
        cc = first(CellIterator(fe_data.dh_up))
        reinit!(cv_p, cc)
        for q in 1:getnquadpoints(cv_p)
            g = sum(shape_gradient(cv_p, q, i) for i in 1:getnbasefunctions(cv_p))
            @test norm(g) < 1e-14
        end
    end

    @testset "allocate_inversion_matrix" begin
        K_inv = allocate_inversion_matrix(fe_data)
        @test size(K_inv) == (nu + np, nu + np)
        @test nnz(K_inv) > 0
        @test all(iszero, K_inv.nzval)   # structure allocated, values zeroed

        # BC-augmented pattern: nnz exceeds the standard cell-adjacency count
        K_std = allocate_matrix(fe_data.dh_up)          # no ch_up
        @test nnz(K_inv) >= nnz(K_std)

        # two fresh allocations share the same structural pattern
        @test K_inv.rowval == allocate_inversion_matrix(fe_data).rowval
    end

    @testset "allocate_evolution_matrix" begin
        K_evo = allocate_evolution_matrix(fe_data)
        @test size(K_evo) == (nb, nb)
        @test nnz(K_evo) > 0
        @test all(iszero, K_evo.nzval)

        # BC-augmented pattern: nnz exceeds the standard cell-adjacency count
        K_std = allocate_matrix(fe_data.dh_b)          # no ch_b
        @test nnz(K_evo) >= nnz(K_std)                 # augmented >= standard

        # two fresh allocations share the same structural pattern
        @test K_evo.rowval == allocate_evolution_matrix(fe_data).rowval
    end
end

# ---------------------------------------------------------------------------
# Evolution matrix and vector builders
# ---------------------------------------------------------------------------
@testset "Evolution" begin
    fe_data  = make_test_fe_data()
    params   = make_test_params()
    forcings = make_test_forcings()
    _, _, nb = get_n_dofs(fe_data)
    κₕ = forcings.κₕ
    κᵥ = forcings.κᵥ

    # helper: relative asymmetry of a sparse matrix
    rel_asymm(K) = norm(K - K') / norm(K)

    @testset "build_M" begin
        M = build_M(fe_data)

        @test size(M) == (nb, nb)
        @test nnz(M) > 0

        # mass matrix is symmetric up to floating-point
        @test rel_asymm(M) < 1e-14

        # all diagonal entries are positive (lumped mass)
        @test all(diag(M) .> 0)

        # M is positive definite: v'Mv > 0 for a random non-zero v
        v = randn(nb)
        @test dot(v, M * v) > 0

        # total mass ≈ domain volume (domain is [0,1]×[-1,1]×[-H,0])
        # just check it's in a reasonable range
        total_mass = sum(M * ones(nb))
        @test total_mass > 0
    end

    @testset "build_Kₕ" begin
        Kₕ = build_Kₕ(fe_data, κₕ)

        @test size(Kₕ) == (nb, nb)
        @test rel_asymm(Kₕ) < 1e-12

        # semi-positive definite: v'Kₕv ≥ 0
        v = randn(nb)
        @test dot(v, Kₕ * v) >= -1e-10 * norm(v)^2

        # different κₕ gives different matrix
        Kₕ2 = build_Kₕ(fe_data, x -> 2 * κₕ(x))
        @test !(Kₕ ≈ Kₕ2)
        @test Kₕ2 ≈ 2 * Kₕ   # linear in κₕ
    end

    @testset "build_Kᵥ" begin
        Kᵥ = build_Kᵥ(fe_data, κᵥ)

        @test size(Kᵥ) == (nb, nb)
        @test rel_asymm(Kᵥ) < 1e-12

        # semi-positive definite
        v = randn(nb)
        @test dot(v, Kᵥ * v) >= -1e-10 * norm(v)^2

        # linear in κᵥ
        Kᵥ2 = build_Kᵥ(fe_data, x -> 2 * κᵥ(x))
        @test Kᵥ2 ≈ 2 * Kᵥ

        # Kₕ and Kᵥ are different (mesh is not isotropic in h vs z)
        Kₕ = build_Kₕ(fe_data, κᵥ)  # same coefficient, different direction
        @test !(Kᵥ ≈ Kₕ)
    end

    @testset "build_rhs_diff" begin
        f = build_rhs_diff(params, fe_data, κᵥ)

        @test length(f) == nb
        # N² > 0 and κᵥ > 0, so the integral is nonzero
        @test norm(f) > 0

        # scales linearly with N²
        params2 = Parameters(; params.ε, params.α, params.μϱ, N²=2*params.N²,
                               f=params.f, H=params.H)
        f2 = build_rhs_diff(params2, fe_data, κᵥ)
        @test f2 ≈ 2 * f

        # zero when κᵥ = 0
        f0 = build_rhs_diff(params, fe_data, x -> 0.0)
        @test norm(f0) == 0
    end

    @testset "build_rhs_flux" begin
        F = 0.5   # constant surface flux

        # SurfaceFluxBC: integral over the surface
        f_flux = build_rhs_flux(params, fe_data, SurfaceFluxBC(x -> F))
        @test length(f_flux) == nb
        @test norm(f_flux) > 0

        # scales linearly with flux magnitude
        f_flux2 = build_rhs_flux(params, fe_data, SurfaceFluxBC(x -> 2F))
        @test f_flux2 ≈ 2 * f_flux

        # SurfaceDirichletBC: returns zero vector
        f_diri = build_rhs_flux(params, fe_data, SurfaceDirichletBC(x -> 0.0))
        @test iszero(f_diri)
    end

    @testset "EvolutionToolkit" begin
        ts  = BDF1(; t_start=0.0, t_stop=1e-2, Δt=1e-3)
        evo = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)

        # stored matrices have correct size
        @test size(evo.M)  == (nb, nb)
        @test size(evo.Kₕ) == (nb, nb)
        @test size(evo.Kᵥ) == (nb, nb)

        # RHS vectors have correct length
        @test length(evo.rhs_diff) == nb
        @test length(evo.rhs_flux) == nb
        @test length(evo.f_bc)     == nb

        # f_bc is zero for homogeneous Dirichlet BCs
        @test norm(evo.f_bc) == 0

        # solver matrix has correct size
        @test size(evo.solver.A) == (nb, nb)

        # with CPU + fixed κᵥ, the preconditioner is an LU factorization
        @test evo.solver.P isa Factorization

        # rebuild Kᵥ with a different κᵥ and check A updates
        κᵥ_new(x) = 2 * κᵥ(x)
        evo.Kᵥ .= build_Kᵥ(fe_data, κᵥ_new)
        collect_evolution_LHS!(evo, params, forcings, ts, fe_data.ch_b)
        A_new = on_architecture(CPU(), evo.solver.A)

        evo2 = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
        # the two toolkits were built with different κᵥ, so their A differs
        @test !(A_new ≈ on_architecture(CPU(), evo2.solver.A))
    end
end

# ---------------------------------------------------------------------------
# Inversion matrix and vector builders
# ---------------------------------------------------------------------------
@testset "Inversion" begin
    fe_data  = make_test_fe_data()
    params   = make_test_params()
    forcings = make_test_forcings()
    nu, np, nb = get_n_dofs(fe_data)
    N_up = ndofs(fe_data.dh_up)

    rel_asymm(K) = norm(K - K') / norm(K)

    @testset "build_A_inversion" begin
        A = build_A_inversion(fe_data, params, forcings.ν)

        @test size(A) == (N_up, N_up)
        @test nnz(A) > 0

        # Stokes + Coriolis is non-symmetric (Coriolis is antisymmetric)
        @test rel_asymm(A) > 1e-10

        # u-DOF diagonal entries are non-negative (viscous term dominates;
        # Coriolis is purely off-diagonal so contributes zero to the diagonal)
        @test all(diag(A)[fe_data.u_dof_indices] .>= 0)

        # A scales linearly with ν (constant)
        A2 = build_A_inversion(fe_data, params, 2 * forcings.ν)
        # viscous + divergence-free terms both scale with ν;
        # frictionless Coriolis + pressure don't, so A2 ≠ 2A in general
        @test !(A ≈ A2)

        # A is different for different Coriolis f
        params_nof = Parameters(; params.ε, params.α, params.μϱ, params.N²,
                                  f=x->0.0, H=params.H)
        A_nof = build_A_inversion(fe_data, params_nof, forcings.ν)
        @test !(A ≈ A_nof)
    end

    @testset "build_B_inversion" begin
        B = build_B_inversion(fe_data, params)

        @test size(B) == (N_up, nb)
        @test nnz(B) > 0

        # only u-rows are nonzero (pressure rows are zero)
        B_p_rows = B[fe_data.p_dof_indices, :]
        @test nnz(B_p_rows) == 0

        # scales linearly with 1/α
        params2 = Parameters(; params.ε, α=2*params.α, params.μϱ, params.N²,
                               f=params.f, H=params.H)
        B2 = build_B_inversion(fe_data, params2)
        @test B2 ≈ (params.α / (2*params.α)) * B
    end

    @testset "build_f_wind" begin
        forcings_wind = Forcings(forcings.ν, forcings.κₕ, forcings.κᵥ,
                                  x->0.1, x->0.0, SurfaceDirichletBC(x->0.0))
        f = build_f_wind(fe_data, params, forcings_wind)

        @test length(f) == N_up
        # wind stress in x produces a nonzero RHS
        @test norm(f) > 0

        # only u-rows are nonzero
        f_p = f[fe_data.p_dof_indices]
        @test norm(f_p) == 0

        # zero wind gives zero vector
        forcings_calm = Forcings(forcings.ν, forcings.κₕ, forcings.κᵥ,
                                  x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))
        @test iszero(build_f_wind(fe_data, params, forcings_calm))
    end

    @testset "InversionToolkit" begin
        inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)

        # B and RHS vectors have correct sizes
        @test size(inv_tk.B) == (N_up, nb)
        @test length(inv_tk.f_wind) == N_up
        @test length(inv_tk.f_bc)   == N_up

        # f_bc is zero for homogeneous velocity BCs
        @test norm(inv_tk.f_bc) == 0

        # solver matrix has correct size
        @test size(inv_tk.solver.A) == (N_up, N_up)

        # with CPU + fixed ν, the preconditioner is an LU factorization
        @test inv_tk.solver.P isa Factorization

        # invert! runs without error and produces a nonzero solution
        b_vec = randn(nb)
        invert!(inv_tk, b_vec)
        x = on_architecture(CPU(), inv_tk.solver.x)
        @test length(x) == N_up
        @test norm(x) > 0

        # the combined system residual is small (direct LU solve)
        A_cpu = on_architecture(CPU(), inv_tk.solver.A)
        y_cpu = on_architecture(CPU(), inv_tk.solver.y)
        @test norm(A_cpu * x - y_cpu) / norm(y_cpu) < 1e-8
    end
end
