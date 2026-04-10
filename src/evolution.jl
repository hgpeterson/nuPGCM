struct EvolutionToolkit{A<:AbstractArchitecture, M, V, S<:IterativeSolverToolkit, I}
# struct EvolutionToolkit{A<:AbstractArchitecture, AS, M, V, VV, S<:IterativeSolverToolkit, I}
    arch::A      # architecture (CPU or GPU)
    # assembler::AS # sparse matrix assembler (Gridap)
    M::M         # Mass matrix
    Kₕ::M        # Horiz. stiffness matrix
    Kᵥ::M        # Vert. stiffness matrix
    # Kᵥ_cache::M  # cache for Kᵥ rebuilds
    rhs_diff::V  # rhs vector from diffusion
    rhs_flux::V  # rhs vector from surface b flux
    rhsₘ::V      # correction vector to add to rhs due to Dirichlet b.c. in M
    rhsₕ::V      # correction vector to add to rhs due to Dirichlet b.c. in Kₕ
    rhsᵥ::V      # correction vector to add to rhs dur to Dirichlet b.c. in Kᵥ
    # rhsᵥ::VV      # correction vector to add to rhs dur to Dirichlet b.c. in Kᵥ
    # rhsᵥ_cache::VV # cache for rhsᵥ rebuilds
    solver::S    # iterative solver toolkit
end

function Base.summary(evolution::EvolutionToolkit)
    t = typeof(evolution)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, evolution::EvolutionToolkit)
    println(io, summary(evolution), ":")
    println(io, "├── arch: ", evolution.arch)
    # println(io, "├── assembler: ", summary(evolution.assembler))
    println(io, "├── M: ", summary(evolution.M))
    println(io, "├── Kₕ: ", summary(evolution.Kₕ))
    println(io, "├── Kᵥ: ", summary(evolution.Kᵥ))
    # println(io, "├── Kᵥ_cache: ", summary(evolution.Kᵥ_cache))
    println(io, "├── rhs_diff: ", summary(evolution.rhs_diff))
    println(io, "├── rhs_flux: ", summary(evolution.rhs_flux))
    println(io, "├── rhsₘ: ", summary(evolution.rhsₘ))
    println(io, "├── rhsₕ: ", summary(evolution.rhsₕ))
    println(io, "├── rhsᵥ: ", summary(evolution.rhsᵥ))
    # println(io, "├── rhsᵥ_cache: ", summary(evolution.rhsᵥ_cache))
      print(io, "└── solver: ", summary(evolution.solver))
end

"""
    evolution_toolkit = EvolutionToolkit(arch::AbstractArchitecture, 
                                         fe_data::FEData, 
                                         params::Parameters, 
                                         forcings::Forcings,
                                         timestepper::AbstractTimestepper; 
                                         kwargs...)
                
Set up the evolution toolkit, which contains the matrices and solvers for the evolution problem.

The PG buoyancy evolution equation is:
```math
μϱ ( ∂ₜb + u·∇b ) = α²ε² [ ∇ₕ·(κₕ∇ₕb) + ∂z(κᵥ∂z b) ].
```
"""
function EvolutionToolkit(arch::AbstractArchitecture, 
                          fe_data::FEData, 
                          params::Parameters, 
                          forcings::Forcings,
                          timestepper::AbstractTimestepper; 
                          atol=1e-6, 
                          rtol=1e-6, 
                          itmax=0, 
                          history=true, 
                          verbose=false)
    # unpack
    B_trial = fe_data.spaces.B_trial
    B_test = fe_data.spaces.B_test
    b_diri = fe_data.spaces.b_diri
    dΩ = fe_data.mesh.dΩ
    κₕ = forcings.κₕ
    κᵥ = forcings.κᵥ

    # build
    @info "Building evolution system..."

    # save sparse assembler for efficient re-building later
    assembler = Gridap.SparseMatrixAssembler(B_trial, B_test)

    # build components
    M, rhsₘ = build_M(B_trial, B_test, dΩ, b_diri)
    Kₕ, rhsₕ = build_Kₕ(B_trial, B_test, dΩ, b_diri, κₕ)
    Kᵥ, rhsᵥ = build_Kᵥ(B_trial, B_test, dΩ, b_diri, κᵥ)
    rhs_diff = build_rhs_diff(params, fe_data, κᵥ)
    rhs_flux = build_rhs_flux(params, forcings, fe_data)

    # # save caches for rebuilds
    # Kᵥ_cache = copy(Kᵥ)
    # rhsᵥ_cache = copy(rhsᵥ)

    # re-order dofs
    perm = fe_data.dofs.p_b
    M = M[perm, perm]
    Kₕ = Kₕ[perm, perm]
    Kᵥ = Kᵥ[perm, perm]
    rhs_diff = rhs_diff[perm]
    rhs_flux = rhs_flux[perm]
    rhsₘ = rhsₘ[perm]
    rhsₕ = rhsₕ[perm]
    rhsᵥ = rhsᵥ[perm]

    # put rhs vectors on GPU if needed
    rhs_diff = on_architecture(arch, rhs_diff)
    rhs_flux = on_architecture(arch, rhs_flux)
    rhsₘ = on_architecture(arch, rhsₘ)
    rhsₕ = on_architecture(arch, rhsₕ)
    rhsᵥ = on_architecture(arch, rhsᵥ)
    # rhsᵥ = on_architecture(arch, rhsᵥ)  # not this one because it gets rebuilt

    # combine to make evolution LHS
    A, P = collect_evolution_LHS(arch, params, forcings, timestepper, M, Kₕ, Kᵥ)

    # rhs vector for solver
    N = size(A, 1)
    T = eltype(A)
    y = on_architecture(arch, zeros(T, N))

    # CG solver
    VT = vector_type(arch, T)
    workspace = Krylov.CgWorkspace(N, N, VT)
    workspace.x .= zero(T)

    # setup solver toolkit
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)
    solver = IterativeSolverToolkit(A, P, y, workspace, kwargs, "Evolution")

    return EvolutionToolkit(arch, M, Kₕ, Kᵥ, rhs_diff, rhs_flux, rhsₘ, rhsₕ, rhsᵥ, solver)
    # return EvolutionToolkit(arch, assembler, M, Kₕ, Kᵥ, Kᵥ_cache, 
    #                         rhs_diff, rhs_flux, rhsₘ, rhsₕ, rhsᵥ, rhsᵥ_cache, solver, order)
end

function collect_evolution_LHS!(evolution::EvolutionToolkit, params::Parameters, forcings::Forcings, timestepper::AbstractTimestepper)
    arch = evolution.arch
    M = evolution.M
    Kₕ = evolution.Kₕ
    Kᵥ = evolution.Kᵥ
    A, P = collect_evolution_LHS(arch, params, forcings, timestepper, M, Kₕ, Kᵥ)
    evolution.solver.A = on_architecture(arch, A)
    evolution.solver.P = P
    return evolution
end
function collect_evolution_LHS(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, timestepper::BDF1, M, Kₕ, Kᵥ)
    θ = evolution_parameter(params, timestepper)
    A = M + θ*(Kₕ + Kᵥ) 

    # preconditioner
    if typeof(arch) == GPU || forcings.conv_param.is_on || timestepper.adaptive
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $(size(A, 1)) DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end

    # move to arch
    A = on_architecture(arch, A)

    return A, P
end
#TODO: Can unify this with BDF1 once adaptive timestepping is implemented for BDF2
function collect_evolution_LHS(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, timestepper::BDF2, M, Kₕ, Kᵥ)
    θ = evolution_parameter(params, timestepper)
    A = M + θ*(Kₕ + Kᵥ) 

    # preconditioner
    if typeof(arch) == GPU || forcings.conv_param.is_on
        P = Diagonal(on_architecture(arch, Vector(1 ./ diag(A))))
    else
        @warn "LU-factoring evolution matrix with $(size(A, 1)) DOFs..."
        @time "lu(A_evol)" P = lu(A)
    end

    # move to arch
    A = on_architecture(arch, A)

    return A, P
end

"""
    θ = evolution_parameter(p::Parameters, timestepper::AbstractTimestepper)

Returns the coefficient needed to build the LHS matrix in the evolution problem of the form
```math
A = M + θ*(Kₕ + Kᵥ)
```
"""
function evolution_parameter(p::Parameters, ts::BDF1)
    return ts.Δt[] * p.α^2 * p.ε^2 / p.μϱ
end
function evolution_parameter(p::Parameters, ts::BDF2)
    #TODO: This assumes fixed Δt.
    return 2/3 * ts.Δt[] * p.α^2 * p.ε^2 / p.μϱ
end

####
#### Matrix-building functions
####

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
function build_matrix_vector!(A, rhs, a, assembler, b_diri)
    Gridap.assemble_matrix!(A, assembler, a)
    Gridap.assemble_vector!(rhs, assembler, d -> a(b_diri, d))
end

# RHS: ∫( ∂z( κᵥ [N² + ∂z(b)] ) d )dΩ
# IBP: ∫( κᵥ [N² + ∂z(b)] d )dΓ - ∫( κᵥ N² ∂z(d) )dΩ - ∫( κᵥ ∂z(b) ∂z(d) )dΩ

function build_rhs_diff(params::Parameters, fe_data::FEData, κᵥ)
    # unpack
    N² = params.N²
    dΩ = fe_data.mesh.dΩ
    B_test = fe_data.spaces.B_test

    # rhs vector for nonzero N²
    l(d) = ∫( -N² * (κᵥ * ∂z(d)) )dΩ
    return assemble_vector(l, B_test)
end

function build_rhs_flux(params::Parameters, forcings::Forcings, fe_data::FEData)
    return build_rhs_flux(params, fe_data, forcings.b_surface_bc)
end
function build_rhs_flux(params::Parameters, fe_data::FEData, bc::SurfaceFluxBC)
    # unpack
    α = params.α
    dΓ = fe_data.mesh.dΓ
    B_test = fe_data.spaces.B_test

    # rhs vector surface buoyancy flux [α²ε²/μϱ κᵥ [N² + ∂z(b)] = α*F]
    l(d) = ∫( α * (bc.flux * d) )dΓ  
    return assemble_vector(l, B_test)
end
function build_rhs_flux(params::Parameters, fe_data::FEData, bc::SurfaceDirichletBC)
    # no flux (Dirichlet b.c.)
    return zeros(fe_data.dofs.nb)
end