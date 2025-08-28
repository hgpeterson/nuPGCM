mutable struct EvolutionToolkit{A<:AbstractArchitecture, M, V, SA<:IterativeSolverToolkit, SHD<:IterativeSolverToolkit, SVD<:IterativeSolverToolkit}
    arch::A           # architecture (CPU or GPU)
    B_hdiff::M        # RHS horizontal diffusion matrix
    b_hdiff::V        # RHS horizontal diffusion vector
    B_vdiff::M        # RHS vertical diffusion matrix
    b_vdiff::V        # RHS vertical diffusion vector
    solver_adv::SA    # advection iterative solver
    solver_hdiff::SHD # horizontal diffusion iterative solver
    solver_vdiff::SVD # vertical diffusion iterative solver
end

function EvolutionToolkit(arch::AbstractArchitecture, fe_data::FEData, params::Parameters, forcings::Forcings; 
                          filename="", force_build=false, kwargs...)
    if isfile(filename) && !force_build
        A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = 
            load_evolution_system(params, filename)
    else
        A_adv, A_hdiff, B_hdiff, b_hdiff, A_vdiff, B_vdiff, b_vdiff = 
            build_evolution_system(fe_data, params, forcings; filename)
    end

    # re-order dofs
    A_adv  =  A_adv[fe_data.dofs.p_b, fe_data.dofs.p_b]
    A_hdiff = A_hdiff[fe_data.dofs.p_b, fe_data.dofs.p_b]
    B_hdiff = B_hdiff[fe_data.dofs.p_b, :]
    b_hdiff = b_hdiff[fe_data.dofs.p_b]
    A_vdiff = A_vdiff[fe_data.dofs.p_b, fe_data.dofs.p_b]
    B_vdiff = B_vdiff[fe_data.dofs.p_b, :]
    b_vdiff = b_vdiff[fe_data.dofs.p_b]

    # preconditioners
    if typeof(arch) == CPU 
        P_hdiff = lu(A_hdiff)
        # P_vdiff = lu(A_vdiff)
        P_vdiff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_vdiff))))
        P_adv   = lu(A_adv)
    else
        P_hdiff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_hdiff))))
        P_vdiff = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_vdiff))))
        P_adv   = Diagonal(on_architecture(arch, Vector(1 ./ diag(A_adv))))
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
                            A_vdiff, P_vdiff, B_vdiff, b_vdiff; kwargs...)
end

function EvolutionToolkit(arch::AbstractArchitecture,
                          A_adv, P_adv, 
                          A_hdiff, P_hdiff, B_hdiff, b_hdiff,
                          A_vdiff, P_vdiff, B_vdiff, b_vdiff;
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
                            solver_vdiff)
end



function evolve_hdiffusion!(evolution::EvolutionToolkit, b)
    # calculate rhs vector
    arch = evolution.arch
    evolution.solver_hdiff.y .= evolution.B_hdiff*on_architecture(arch, b.free_values) + evolution.b_hdiff

    # solve
    iterative_solve!(evolution.solver_hdiff)

    return evolution
end
