struct EvolutionToolkit{A<:AbstractArchitecture, M, V, S<:IterativeSolverToolkit, I}
    arch::A      # architecture (CPU or GPU)
    M::M         # Mass matrix
    Kₕ::M        # Horiz. stiffness matrix
    Kᵥ::M        # Vert. stiffness matrix
    rhs_diff::V  # rhs vector from diffusion
    rhs_flux::V  # rhs vector from surface b flux
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
    println(io, "├── Kᵥ: ", summary(evolution.Kᵥ))
    println(io, "├── rhs_diff: ", summary(evolution.rhs_diff))
    println(io, "├── rhs_flux: ", summary(evolution.rhs_flux))
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
    if order != 1 && order != 2
        throw(ArgumentError("order $order not yet implemented"))
    end

    # build
    @info "Building evolution system..."
    @time "build evolution system" M, Kₕ, Kᵥ, rhs_diff, rhs_flux, rhsₘ, rhsₕ, rhsᵥ = 
                                        build_evolution_system(fe_data, params, forcings)

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

    # combine to make evolution LHS
    A, P = collect_evolution_LHS(arch, params, forcings, M, Kₕ, Kᵥ, order)

    # rhs vector for solver
    N = size(A, 1)
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

    return EvolutionToolkit(arch, M, Kₕ, Kᵥ, rhs_diff, rhs_flux, rhsₘ, rhsₕ, rhsᵥ, solver, order)
end

function collect_evolution_LHS!(evolution::EvolutionToolkit, params::Parameters, forcings::Forcings)
    arch = evolution.arch
    M = evolution.M
    Kₕ = evolution.Kₕ
    Kᵥ = evolution.Kᵥ
    order = evolution.order
    A, P = collect_evolution_LHS(arch, params, forcings::Forcings, M, Kₕ, Kᵥ, order)
    evolution.solver.A = on_architecture(arch, A)
    evolution.solver.P = P
    return evolution
end
function collect_evolution_LHS(arch::AbstractArchitecture, params::Parameters, forcings::Forcings, M, Kₕ, Kᵥ, order)
    θ = evolution_parameter(params, Val(order))
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
    θ = evolution_parameter(p::Parameters, order::Val)

Returns the coefficient needed to build the LHS matrix in the evolution problem of the form
```math
A = M + θ*(Kₕ + Kᵥ)
```
For `order` = 1, we use Backwards Euler (BDF1), so 
```math
θ = Δt α² ε² / μϱ.
```
For `order` = 2, we use BDF2:
```math
θ = 2/3 Δt α² ε² / μϱ.
```
"""
function evolution_parameter(p::Parameters, ::Val{1})
    # BDF1
    return p.Δt * p.α^2 * p.ε^2 / p.μϱ
end
function evolution_parameter(p::Parameters, ::Val{2})
    # BDF2
    return 2/3 * p.Δt * p.α^2 * p.ε^2 / p.μϱ
end

####
#### Matrix-building functions
####

"""
    M, Kₕ, Kᵥ, rhs_diff, rhsₘ, rhsₕ, rhsᵥ = 
build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings)

Build the matrices for the evolution problem of the PG equations.

The evolution equation is written as
```math
μϱ ( ∂ₜb + u·∇b ) = α²ε² [ ∇ₕ·(κₕ∇ₕb) + ∂z(κᵥ∂z b) ]
```

See also [`build_M`](@ref), [`build_Kₕ`](@ref), [`build_Kᵥ`](@ref).
"""
function build_evolution_system(fe_data::FEData, params::Parameters, forcings::Forcings)
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
    rhs_diff = build_rhs_diff(params, fe_data, κᵥ)
    rhs_flux = build_rhs_flux(params, forcings, fe_data)
    return M, Kₕ, Kᵥ, rhs_diff, rhs_flux, rhsₘ, rhsₕ, rhsᵥ
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
    Δt = params.Δt
    dΓ = fe_data.mesh.dΓ
    B_test = fe_data.spaces.B_test

    # rhs vector surface buoyancy flux [α²ε²/μϱ κᵥ [N² + ∂z(b)] = α*F]
    l(d) = ∫( Δt * α * (bc.flux * d) )dΓ  
    return assemble_vector(l, B_test)
end
function build_rhs_flux(params::Parameters, fe_data::FEData, bc::SurfaceDirichletBC)
    # no flux (Dirichlet b.c.)
    return zeros(fe_data.dofs.nb)
end