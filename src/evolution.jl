struct EvolutionToolkit{A<:AbstractArchitecture, M, V, S<:IterativeSolverToolkit, I}
    arch::A      # architecture (CPU or GPU)
    M::M         # Mass matrix
    Kₕ::M        # Horiz. stiffness matrix
    rhs_diff::V  # rhs vector from diffusion
    rhsₘ::V      # correction vector to add to rhs due to Dirichlet b.c. in M
    rhsₕ::V      # correction vector to add to rhs due to Dirichlet b.c. in Kₕ
    rhsᵥ::V      # correction vector to add to rhs dur to Dirichlet b.c. in Kᵥ
    solver::S    # iterative solver toolkit
    order::I     # order of timestepping scheme
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
    println(io, "├── rhs_diff: ", summary(evolution.rhs_diff))
    println(io, "├── rhsₘ: ", summary(evolution.rhsₘ))
    println(io, "├── rhsₕ: ", summary(evolution.rhsₕ))
    println(io, "├── rhsᵥ: ", summary(evolution.rhsᵥ))
    println(io, "├── solver: ", summary(evolution.solver))
      print(io, "└── order: ", evolution.order)
end

"""
    evolution_toolkit = EvolutionToolkit(arch::AbstractArchitecture, 
                                         fe_data::FEData, 
                                         params::Parameters, 
                                         forcings::Forcings; 
                                         kwargs...)
                
Set up the evolution toolkit, which contains the matrices and solvers for the evolution problem.
"""
function EvolutionToolkit(arch::AbstractArchitecture, 
                          fe_data::FEData, 
                          params::Parameters, 
                          forcings::Forcings; 
                          order=2,
                          atol=1e-6, 
                          rtol=1e-6, 
                          itmax=0, 
                          history=true, 
                          verbose=false)
    if order != 1
        throw(ArgumentError("order $order not yet implemented"))
    end

    # build
    @info "Building evolution system..."
    M, Kₕ, Kᵥ, rhs_diff, rhsₘ, rhsₕ, rhsᵥ = build_evolution_system(fe_data, params, forcings; order)

    # re-order dofs
    perm = fe_data.dofs.p_b
    M = M[perm, perm]
    Kₕ = Kₕ[perm, perm]
    Kᵥ = Kᵥ[perm, perm]
    rhs_diff = rhs_diff[perm]
    rhsₘ = rhsₘ[perm]
    rhsₕ = rhsₕ[perm]
    rhsᵥ = rhsᵥ[perm]

    # combine to make evolution LHS
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    θ = Δt * α^2 * ε^2 / μϱ
    A = M + θ*(Kₕ + Kᵥ) 
    N = size(A, 1)

    # preconditioner
    if typeof(arch) == GPU || forcings.conv_param.is_on
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $N DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end

    # move to arch
    A = on_architecture(arch, A)

    # rhs vector for solver
    T = eltype(A)
    y = on_architecture(arch, zeros(T, N))

    # CG solver
    VT = vector_type(arch, T)
    solver = Krylov.CgSolver(N, N, VT)
    solver.x .= zero(T)

    # setup solver toolkit
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)
    solver = IterativeSolverToolkit(A, P, y, solver, kwargs, "Evolution")

    return EvolutionToolkit(arch, M, Kₕ, rhs_diff, rhsₘ, rhsₕ, rhsᵥ, solver, order)
end

"""
    A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = 
        build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings)

Build the matrices for the evolution problem of the PG equations.

The evolution equation is written as
```math
μϱ ( ∂ₜb + u·∇b ) = α²ε² [ ∇ₕ·(κₕ∇ₕb) + ∂z(κᵥ∂z b) ]
```

See also [`build_M`](@ref), [`build_Kₕ`](@ref), [`build_Kᵥ`](@ref).
"""
function build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings; order)
    if order != 1
        throw(ArgumentError("order $order not yet implemented"))
    end

    @time "build evolution system" begin

    # unpack
    B_trial = fe_data.spaces.B_trial
    B_test = fe_data.spaces.B_test
    b_diri = fe_data.spaces.b_diri
    dΩ = fe_data.mesh.dΩ
    κₕ = forcings.κₕ
    κᵥ = forcings.κᵥ

    # build components
    M, rhsₘ = build_M(B_trial, B_test, dΩ, b_diri)
    Kₕ, rhsₕ = build_Kₕ(B_trial, B_test, dΩ, b_diri, κₕ)
    Kᵥ, rhsᵥ = build_Kᵥ(B_trial, B_test, dΩ, b_diri, κᵥ)
    rhs_diff = build_rhs_diff(params, forcings, fe_data, κᵥ)

    end
    return M, Kₕ, Kᵥ, rhs_diff, rhsₘ, rhsₕ, rhsᵥ
end

"""
    M, rhsₘ = build_M(B, D, dΩ, b_diri)

Build FE mass matrix `M` and right-hand-side vector `rhsₘ` correction due to Dirichlet b.c.

`B` and `D` are TrialFESpace and TestFESpace's, respectively. `dΩ` is the Measure, and b_diri is a FEFunction that is 
equal to the Dirichlet b.c. on the boundary and zero elsewhere.

See also [`build_matrix_vector`](@ref).
"""
function build_M(B, D, dΩ, b_diri)
    aₘ(b, d) = ∫( b*d )dΩ
    return build_matrix_vector(aₘ, B, D, b_diri)
end

"""
    Kₕ, rhsₕ = build_Kₕ(B, D, dΩ, b_diri, κₕ)

Build FE stiffness matrix `Kₕ` and right-hand-side vector `rhsₕ` correction due to Dirichlet b.c.

`B` and `D` are TrialFESpace and TestFESpace's, respectively. `dΩ` is the Measure, and b_diri is a FEFunction that is 
equal to the Dirichlet b.c. on the boundary and zero elsewhere. `κₕ` is the horizontal diffusivity.

See also [`build_matrix_vector`](@ref).
"""
function build_Kₕ(B, D, dΩ, b_diri, κₕ)
    aₕ(b, d) = ∫( κₕ*(∂x(b)*∂x(d) + ∂y(b)*∂y(d)) )dΩ
    return build_matrix_vector(aₕ, B, D, b_diri)
end

"""
    Kᵥ, rhsᵥ = build_Kᵥ(fe_data::FEData, κᵥ)
    Kᵥ, rhsᵥ = build_Kᵥ(B, D, dΩ, b_diri, κᵥ)

Build FE stiffness matrix `Kᵥ` and right-hand-side vector `rhsᵥ` correction due to Dirichlet b.c.

`B` and `D` are TrialFESpace and TestFESpace's, respectively. `dΩ` is the Measure, and b_diri is a FEFunction that is 
equal to the Dirichlet b.c. on the boundary and zero elsewhere. `κᵥ` is the vertical diffusivity.

See also [`build_matrix_vector`](@ref).
"""
function build_Kᵥ(fe_data::FEData, κᵥ)
    return build_Kᵥ(fe_data.spaces.B_trial, fe_data.spaces.B_test, fe_data.mesh.dΩ, fe_data.spaces.b_diri, κᵥ)
end
function build_Kᵥ(B, D, dΩ, b_diri, κᵥ)
    aᵥ(b, d) = ∫( κᵥ*∂z(b)*∂z(d) )dΩ
    return build_matrix_vector(aᵥ, B, D, b_diri)
end

"""
    A, rhs = build_matrix_vector(a, B, D, b_diri)

Build FE matrix `A` and right-hand-side vector `rhs` correction due to Dirichlet b.c.

`a` defines the bilinear form. `B` and `D` are TrialFESpace and TestFESpace's, respectively. b_diri is a FEFunction that 
is  equal to the Dirichlet b.c. on the boundary and zero elsewhere.
"""
function build_matrix_vector(a, B, D, b_diri)
    A = assemble_matrix(a, B, D)
    rhs = assemble_vector(d -> a(b_diri, d), D)
    return A, rhs
end

function build_rhs_diff(params::Parameters, forcings::Forcings, fe_data::FEData, κᵥ)
    return build_rhs_diff(params, fe_data, forcings.b_surface_bc, κᵥ)
end
function build_rhs_diff(params::Parameters, fe_data::FEData, bc::SurfaceFluxBC, κᵥ)
    # unpack
    N² = params.N²
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    dΩ = fe_data.mesh.dΩ
    dΓ = fe_data.mesh.dΓ
    B_test = fe_data.spaces.B_test

    # RHS: ∫( ∂z( κᵥ [N² + ∂z(b)] ) d )dΩ
    # IBP: ∫( κᵥ [N² + ∂z(b)] d )dΓ - ∫( κᵥ N² ∂z(d) )dΩ - ∫( κᵥ ∂z(b) ∂z(d) )dΩ

    # rhs vector for nonzero N² and surface buoyancy flux [α²ε²/μϱ κᵥ [N² + ∂z(b)] = α*F]
    θ = Δt * α^2 * ε^2 / μϱ
    l(d) = ∫( -θ * N² * κᵥ * ∂z(d) )dΩ + ∫( Δt*α*(bc.flux*d) )dΓ  
    return assemble_vector(l, B_test)
end
function build_rhs_diff(params::Parameters, fe_data::FEData, bc::SurfaceDirichletBC, κᵥ)
    # unpack
    N² = params.N²
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    dΩ = fe_data.mesh.dΩ
    B_test = fe_data.spaces.B_test

    # rhs vector for nonzero N² (no surface buoyancy flux)
    θ = Δt * α^2 * ε^2 / μϱ
    l(d) = ∫( -θ * N² * κᵥ * ∂z(d) )dΩ
    return assemble_vector(l, B_test)
end