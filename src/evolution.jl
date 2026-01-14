mutable struct EvolutionToolkit{A<:AbstractArchitecture, M, V, SA<:IterativeSolverToolkit, 
                                SHD<:IterativeSolverToolkit, SVD<:IterativeSolverToolkit, I}
    arch::A            # architecture (CPU or GPU)
    B_hdiff::M         # RHS horizontal diffusion matrix
    b_hdiff::V         # RHS horizontal diffusion vector
    B_vdiff::M         # RHS vertical diffusion matrix
    b_vdiff::V         # RHS vertical diffusion vector
    solver_adv::SA     # advection iterative solver
    solver_hdiff::SHD  # horizontal diffusion iterative solver
    solver_vdiff::SVD  # vertical diffusion iterative solver
    order::I           # order of timestepping scheme
end

function Base.summary(evolution::EvolutionToolkit)
    t = typeof(evolution)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, evolution::EvolutionToolkit)
    println(io, summary(evolution), ":")
    println(io, "├── arch: ", evolution.arch)
    println(io, "├── B_hdiff: ", summary(evolution.B_hdiff))
    println(io, "├── b_hdiff: ", summary(evolution.b_hdiff))
    println(io, "├── B_vdiff: ", summary(evolution.B_vdiff))
    println(io, "├── b_vdiff: ", summary(evolution.b_vdiff))
    println(io, "├── solver_adv: ", summary(evolution.solver_adv))
    println(io, "├── solver_hdiff: ", summary(evolution.solver_hdiff))
    println(io, "├── solver_vdiff: ", summary(evolution.solver_vdiff))
      print(io, "└── order: ", evolution.order)
end

"""
    evolution_toolkit = EvolutionToolkit(arch::AbstractArchitecture, 
                                         fe_data::FEData, 
                                         params::Parameters, 
                                         forcings::Forcings; 
                                         order=2,
                                         kwargs...)
                
Set up the evolution toolkit, which contains the matrices and solvers for the evolution problem.
"""
function EvolutionToolkit(arch::AbstractArchitecture, 
                          fe_data::FEData, 
                          params::Parameters, 
                          forcings::Forcings; 
                          order=2,
                          kwargs...)
    # build
    @info "Building evolution system..."
    A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = 
        build_evolution_system(fe_data, params, forcings; order)

    # re-order dofs
    A_adv  =  A_adv[fe_data.dofs.p_b, fe_data.dofs.p_b]
    A_hdiff = A_hdiff[fe_data.dofs.p_b, fe_data.dofs.p_b]
    B_hdiff = B_hdiff[fe_data.dofs.p_b, :]
    b_hdiff = b_hdiff[fe_data.dofs.p_b]
    A_vdiff = A_vdiff[fe_data.dofs.p_b, fe_data.dofs.p_b]
    B_vdiff = B_vdiff[fe_data.dofs.p_b, :]
    b_vdiff = b_vdiff[fe_data.dofs.p_b]

    # preconditioners
    if typeof(arch) == GPU 
        P_hdiff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_hdiff))))
        P_adv   = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_adv))))
        P_vdiff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_vdiff))))
    else
        @warn "LU-factoring evolution matrices with $(fe_data.dofs.nb) DOFs..."
        @time "lu(A_adv)" P_adv = lu(A_adv)
        @time "lu(A_hdiff)" P_hdiff = lu(A_hdiff)
        if forcings.conv_param.is_on
            # vertical diffusion matrix will get rebuilt for convection, so use diagonal preconditioner
            P_vdiff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_vdiff))))
        else
            @time "lu(A_vdiff)" P_vdiff = lu(A_vdiff)
        end
    end

    # move to arch
    A_adv   = on_architecture(arch, A_adv)
    A_hdiff = on_architecture(arch, A_hdiff)
    B_hdiff = on_architecture(arch, B_hdiff)
    b_hdiff = on_architecture(arch, b_hdiff)
    A_vdiff = on_architecture(arch, A_vdiff)
    B_vdiff = on_architecture(arch, B_vdiff)
    b_vdiff = on_architecture(arch, b_vdiff)

    return EvolutionToolkit(arch, 
                            A_adv, P_adv, 
                            A_hdiff, P_hdiff, B_hdiff, b_hdiff, 
                            A_vdiff, P_vdiff, B_vdiff, b_vdiff; order, kwargs...)
end

function EvolutionToolkit(arch::AbstractArchitecture,
                          A_adv, P_adv, 
                          A_hdiff, P_hdiff, B_hdiff, b_hdiff,
                          A_vdiff, P_vdiff, B_vdiff, b_vdiff;
                          order,
                          atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=false)
    # rhs vector for solvers
    N = size(A_adv, 1)
    T = eltype(A_adv)
    y = on_architecture(arch, zeros(T, N))

    # use CG solver for all three systems
    VT = vector_type(arch, T)
    solver = Krylov.CgSolver(N, N, VT)
    solver.x .= zero(T)

    # setup key word arguments for solver toolkit
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)

    # each system gets its own solver toolkit
    solver_adv = IterativeSolverToolkit(A_adv, P_adv, y, solver, kwargs, "Advection")
    solver_hdiff = IterativeSolverToolkit(A_hdiff, P_hdiff, y, solver, kwargs, "Horizontal Diffusion")
    solver_vdiff = IterativeSolverToolkit(A_vdiff, P_vdiff, y, solver, kwargs, "Vertical Diffusion")

    return EvolutionToolkit(arch,
                            B_hdiff, b_hdiff, 
                            B_vdiff, b_vdiff, 
                            solver_adv, 
                            solver_hdiff, 
                            solver_vdiff,
                            order)
end

"""
    evolution = evolve_advection!(evolution::EvolutionToolkit, b)

Perform horizontal diffusion part of one evolution step given buoyancy `b`.
"""
function evolve_hdiffusion!(evolution::EvolutionToolkit, b)
    # calculate rhs vector
    arch = evolution.arch
    evolution.solver_hdiff.y .= evolution.B_hdiff*on_architecture(arch, b.free_values) + evolution.b_hdiff

    # solve
    iterative_solve!(evolution.solver_hdiff)

    return evolution
end