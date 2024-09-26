"""
    geometry = Geometry(mesh, dim, hmin, hmax, Ω, dΩ)

A geometry object that contains the `mesh`, dimension `dim`, minimum and maximum 
line segment lengths `hmin` and `hmax`, triangulation `Ω`, and integration 
measure `dΩ`.
"""
struct Geometry{M, D, H, T, dT}
    mesh::M
    dim::D
    hmin::H
    hmax::H
    Ω::T
    dΩ::dT
end

"""
    geometry = Geometry(mesh_file::AbstractString)

Create a geometry object from a Gmsh .msh file `mesh_file`.
"""
function Geometry(mesh_file::AbstractString)
    # load mesh
    mesh = GmshDiscreteModel(mesh_file)

    # dimension
    if length(mesh.grid.node_coordinates[1]) == 3
        dim = TwoD()
    elseif length(mesh.grid.node_coordinates[1]) == 4
        dim = ThreeD()
    end

    # mesh resolution
    hmin, hmax = resolution(mesh.grid)

    # triangulation and integration measure
    Ω = Triangulation(mesh)
    dΩ = Measure(Ω, 4)

    return Geometry(mesh, dim, hmin, hmax, Ω, dΩ)
end

"""
    hmin, hmax = resolution(grid::Gridap.Geometry.Grid)

Compute the minimum and maximum line segment lengths `hmin` and `hmax` of the
`grid`.
"""
function resolution(grid::Gridap.Geometry.Grid)
    p = grid.node_coordinates
    t = grid.cell_node_ids
    h = [[norm(p[t[i][1]] - p[t[i][j]]) for j ∈ 1:length(t[i])] i ∈ 1:length(t)]
    h = reduce(vcat, h) # flatten
    hmin = minimum(h)
    hmax = maximum(h)
    @printf("\nMesh resolution: .2e ≤ h ≤ %.2e\n", hmin, hmax)
    return hmin, hmax
end

################################################################################

"""
    forcing = Forcing(ν, κ)

Struct containing forcings for the problem. `ν` is the turbulent viscosity and
`κ` is the turbulent diffusivity.
"""
struct Forcing{N, K}
    ν::N
    κ::K
end

################################################################################

"""
    parameters = Parameters(ε², γ, f₀, β, μϱ, Δt, T, α, tol, itmax)

Struct containing parameters for the problem. 

    • ε²: Ekman number
    • γ: squared aspect ratio
    • f₀: Coriolis parameter
    • β: Coriolis parameter
    • μϱ: Prandtl number times Burger number
    • Δt: time step
    • T: final time
    • α: for timestepping
    • tol: tolerance for iterative solvers
    • itmax: maximum number of iterations for iterative solvers
"""
struct Parameters{FT, IN} where {FT<:AbstractFloat, IN<:Integer}
    ε²::FT
    γ::FT
    f₀::FT
    β::FT
    μϱ::FT
    Δt::FT
    T::FT
    α::FT
    tol::FT
    itmax::IN
end

"""
    parameters = Parameters(ε², γ, f₀, β, μϱ, Δt, T, tol, itmax)

Create a parameters struct with `α = Δt/2*ε²/μϱ`.
"""
function Parameters(ε², γ, f₀, β, μϱ, Δt, T, tol, itmax)
    α = Δt/2*ε²/μϱ

    println("\nParameters:\n")
    @printf("  ε² = %.1e (δ = %.2e))\n", ε², √(2ε²))
    @printf("  f₀ = %.1e\n", f₀)
    @printf("   β = %.1e\n", β)
    @printf("   γ = %.1e\n", γ)
    @printf("  μϱ = %.1e\n", μϱ)
    @printf("  Δt = %.1e\n", Δt)
    @printf("   T = %.1e\n\n", T)

    return Parameters(ε², γ, f₀, β, μϱ, Δt, T, α, tol, itmax)
end

################################################################################

struct IterativeSolvers{M, V, SI, SE}
    LHS_inversion::M
    perm_inversion::V
    inv_perm_inversion::V
    RHS_inversion::M
    solver_inversion::SI
    LHS_evolution::M
    perm_evolution::V
    inv_perm_evolution::V
    solver_evolution::SE
end

function IterativeSolvers(arch::AbstractArchitecture, geo::Geometry, sp::Spaces, p::Parameters, forcing::Forcing,
                          matrix_folder::AbstractString, mesh_file::AbstractString)
    LHS_inversion, perm_inversion, inv_perm_inversion, RHS_inversion, solver_inversion = setup_inversion(arch, geo, sp, p, forcing, matrix_folder, mesh_file)
    LHS_evolution, perm_evolution, inv_perm_evolution, solver_evolution = setup_evolution(arch, geo, sp, p, forcing, matrix_folder, mesh_file)
    return IterativeSolvers(LHS_inversion, perm_inversion, inv_perm_inversion, RHS_inversion, solver_inversion, LHS_evolution, perm_evolution, inv_perm_evolution, solver_evolution)
end
function setup_inversion(arch::AbstractArchitecture, geo::Geometry, sp::Spaces, p::Parameters, forcing::Forcing, 
                         matrix_folder::AbstractString, mesh_file::AbstractString)
    # unpack 
    dim = geo.dim
    dΩ = geo.dΩ
    X = sp.X
    Y = sp.Y
    B = sp.B
    N = sp.N
    ε² = p.ε²
    γ = p.γ
    f₀ = p.f₀
    β = p.β
    ν = forcing.ν

    # load or build inversion LHS
    LHS_inversion_fname = @sprintf("%s/LHS_inversion_%s_%e_%e_%e_%e.h5", matrix_folder, mesh_file[1:end-4], ε², γ, f₀, β)
    if isfile(LHS_inversion_fname)
        LHS_inversion, perm_inversion, inv_perm_inversion = read_sparse_matrix(LHS_inversion_fname)
    else
        f(x) = f₀ + β*x[2]
        LHS_inversion, perm_inversion, inv_perm_inversion = assemble_LHS_inversion(arch, dim, γ, ε², ν, f, X, Y, dΩ; fname=LHS_inversion_fname)
    end

    # build inversion RHS
    RHS_inversion = assemble_RHS_inversion(perm_inversion, B, Y, dΩ)

    # move to architecture
    LHS_inversion = on_architecture(arch, LHS_inversion)
    RHS_inversion = on_architecture(arch, RHS_inversion)

    # Krylov solver for inversion
    solver_inversion = GmresSolver(N, N, 20, vector_type(arch))

    return LHS_inversion, perm_inversion, inv_perm_inversion, RHS_inversion, solver_inversion
end
function setup_evolution(arch::AbstractArchitecture, geo::Geometry, sp::Spaces, p::Parameters, forcing::Forcing, 
                         matrix_folder::AbstractString, mesh_file::AbstractString)
    # unpack 
    dim = geo.dim
    dΩ = geo.dΩ
    α = p.α
    γ = p.γ
    κ = forcing.κ
    B = sp.B
    D = sp.D
    nb = sp.nb

    # load or build evolution LHS
    LHS_evolution_fname = @sprintf("%s/LHS_evolution_%s_%e_%e.h5", matrix_folder, mesh_file[1:end-4], α, γ)
    if isfile(LHS_evolution_fname)
        LHS_evolution, perm_evolution, inv_perm_evolution  = read_sparse_matrix(LHS_evolution_fname)
    else
        LHS_evolution, perm_evolution, inv_perm_evolution = assemble_LHS_evolution(arch, dim, α, γ, κ, B, D, dΩ; fname=LHS_evolution_fname)
    end

    # preconditioner of 1/diag(A) works well enough for this problem
    P_evolution = Diagonal(Vector(1 ./ diag(LHS_evolution)))

    # move to architecture
    LHS_evolution = on_architecture(arch, LHS_evolution)
    P_evolution = Diagonal(on_architecture(arch, diag(P_evolution)))

    # Krylov solver for evolution
    solver_evolution = CgSolver(nb, nb, vector_type(arch))

    return LHS_evolution, perm_evolution, inv_perm_evolution, solver_evolution
end


"""
    model = Model(dim, arch, geo, forcing, parameters, matix_folder)

Model struct. `dim` is the dimension, `arch` is the architecture, `geo` is the geometry, `forcing` is the forcing, `parameters` is
the parameters, and `matix_folder` is the folder where LHS matrices are stored.
"""
struct Model{D, A, G, F, P}
    dim::D
    arch::A
    geo::G
    forcing::F
    parameters::P
    matix_folder::AbstractString
    state::Union{Nothing, Tuple{Vector{Float64}, Float64}}
end

"""
    set_state!(model, b, t)

Set the initial condition of the model `model` to `b` at time `t`.
"""
function set_state!(model::Model, b::Function, t::Float64)
    # set state
    model.state = (b(model.geo.mesh.node_coordinates), t)
end

"""
    set_state!(model, state_file)

Set the initial condition of the model `model` to the state in `state_file`.
"""
function set_state!(model::Model, state_file::AbstractString)
    # load state
    state = load_state(state_file)
    model.state = state
end

"""
    save_state(ux, uy, uz, p, b, t; fname="state.h5")

Save the state of the model to a file `fname`.
"""
function save_state(ux, uy, uz, p, b, t; fname="state.h5")
    # save state
    h5open(fname, "w") do file
        write(file, "ux", ux)
        write(file, "uy", uy)
        write(file, "uz", uz)
        write(file, "p", p)
        write(file, "b", b)
        write(file, "t", t)
    end
end

"""
    ux, uy, uz, p, b, t = load_state(fname)

Load the state of the model from a file `fname`.
"""
function load_state(fname::AbstractString)
    # load state
    ux = read(fname, "ux")
    uy = read(fname, "uy")
    uz = read(fname, "uz")
    p = read(fname, "p")
    b = read(fname, "b")
    t = read(fname, "t")
    return ux, uy, uz, p, b, t
end

"""
    ux, uy, uz, p, b, t = load_state(file)

Load the state of the model from a file `file`.
"""
function load_state(file::HDF5File)
    # load state
    ux = read(file, "ux")
    uy = read(file, "uy")
    uz = read(file, "uz")
    p = read(file, "p")
    b = read(file, "b")
    t = read(file, "t")
    return ux, uy, uz, p, b, t
end