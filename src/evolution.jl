struct EvolutionToolkit{A<:AbstractArchitecture, M, V, S<:IterativeSolverToolkit, I}
    arch::A      # architecture (CPU or GPU)
    M::M         # Mass matrix
    Kₕ::M        # Horiz. stiffness matrix
    rhs::V       # correction vector to add to rhs
    solver::S    # iterative solver toolkit
    order::I     # order of timestepping scheme
end

# function Base.summary(evolution::EvolutionToolkit)
#     t = typeof(evolution)
#     return "$(parentmodule(t)).$(nameof(t))"
# end
# function Base.show(io::IO, evolution::EvolutionToolkit)
#     println(io, summary(evolution), ":")
#     println(io, "├── arch: ", evolution.arch)
#     println(io, "├── B_hdiff: ", summary(evolution.B_hdiff))
#     println(io, "├── b_hdiff: ", summary(evolution.b_hdiff))
#     println(io, "├── B_vdiff: ", summary(evolution.B_vdiff))
#     println(io, "├── b_vdiff: ", summary(evolution.b_vdiff))
#     println(io, "├── solver_adv: ", summary(evolution.solver_adv))
#     println(io, "├── solver_hdiff: ", summary(evolution.solver_hdiff))
#     println(io, "├── solver_vdiff: ", summary(evolution.solver_vdiff))
#       print(io, "└── order: ", evolution.order)
# end

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
    M, Kₕ, Kᵥ, rhs = build_evolution_system(fe_data, params, forcings; order)

    # re-order dofs
    perm = fe_data.dofs.p_b
    M = M[perm, perm]
    Kₕ = Kₕ[perm, perm]
    Kᵥ = Kᵥ[perm, perm]
    rhs = rhs[perm]

    # combine to make evolution LHS
    ε = params.ε
    α = params.α
    μϱ = params.μϱ
    Δt = params.Δt
    θ = Δt * α^2 * ε^2 / μϱ
    A = M - θ*(Kₕ + Kᵥ) 
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
    M = on_architecture(arch, M)
    Kₕ = on_architecture(arch, Kₕ)
    rhs = on_architecture(arch, rhs)

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

    return EvolutionToolkit(arch, M, Kₕ, rhs, solver, order)
end