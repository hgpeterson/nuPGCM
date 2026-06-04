struct EvolutionToolkit{A<:AbstractArchitecture, M, V, S<:IterativeSolverToolkit}
    arch::A
    M::M         # mass matrix (raw, no BCs)
    Kₕ::M        # horizontal stiffness (raw)
    Kᵥ::M        # vertical stiffness (raw; rebuilt each step when conv_param is on)
    rhs_diff::V  # -N² κᵥ ∂z(b) source term
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
    println(io, "├── M: ", summary(evolution.M))
    println(io, "├── Kₕ: ", summary(evolution.Kₕ))
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
    Kᵥ       = build_Kᵥ(fe_data, κᵥ)
    rhs_diff = build_rhs_diff(params, fe_data, κᵥ)
    rhs_flux = build_rhs_flux(params, forcings, fe_data)

    ts1 = BDF1(; ts.t_start, t_stop=ts.t_stop, Δt=ts.Δt[])
    A, P, f_bc = collect_evolution_LHS(arch, params, forcings, ts1, M, Kₕ, Kᵥ, fe_data.ch_b)

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

    return EvolutionToolkit(arch, M, Kₕ, Kᵥ, rhs_diff, rhs_flux, f_bc, solver)
end

function collect_evolution_LHS!(evolution::EvolutionToolkit, params::Parameters,
                                 forcings::Forcings, ts::AbstractTimestepper, ch_b)
    arch = evolution.arch
    A, P, f_bc = collect_evolution_LHS(arch, params, forcings, ts,
                                        evolution.M, evolution.Kₕ, evolution.Kᵥ, ch_b)
    evolution.solver.A = on_architecture(arch, A)
    evolution.solver.P = P
    evolution.f_bc    .= on_architecture(arch, f_bc)
    return evolution
end

function collect_evolution_LHS(arch::AbstractArchitecture, params::Parameters,
                                forcings::Forcings, ts::BDF1, M, Kₕ, Kᵥ, ch_b)
    θ = evolution_parameter(params, ts)
    A, f_bc = _form_evolution_lhs(θ, M, Kₕ, Kᵥ, ch_b)

    if typeof(arch) == GPU || forcings.conv_param.is_on || ts.adaptive
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $(size(A, 1)) DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end

    A = on_architecture(arch, A)
    return A, P, f_bc
end
function collect_evolution_LHS(arch::AbstractArchitecture, params::Parameters,
                                forcings::Forcings, ts::BDF2, M, Kₕ, Kᵥ, ch_b)
    θ = evolution_parameter(params, ts)
    A, f_bc = _form_evolution_lhs(θ, M, Kₕ, Kᵥ, ch_b)

    if typeof(arch) == GPU || forcings.conv_param.is_on
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $(size(A, 1)) DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end

    A = on_architecture(arch, A)
    return A, P, f_bc
end

function _form_evolution_lhs(θ, M, Kₕ, Kᵥ, ch_b)
    # Start from copy(M) to preserve the BC-augmented sparsity pattern.
    # M + θ*(Kₕ + Kᵥ) via Julia's sparse + drops structural zeros, breaking apply!.
    A = copy(M)
    @. A.nzval = M.nzval + θ * (Kₕ.nzval + Kᵥ.nzval)
    f_bc = zeros(size(A, 1))
    apply!(A, f_bc, ch_b)
    return A, f_bc
end

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
