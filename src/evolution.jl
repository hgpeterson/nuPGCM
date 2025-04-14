struct EvolutionToolkit{A, V, SA<:IterativeSolverToolkit, SD<:IterativeSolverToolkit}
    B_diff::A       # RHS diffusion matrix
    b_diff::V       # RHS diffusion vector
    solver_adv::SA  # advection iterative solver
    solver_diff::SD # diffusion iterative solver
end

function EvolutionToolkit(A_adv, P_adv, A_diff, P_diff, B_diff, b_diff;
                          atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=false)
    arch = architecture(A_adv)
    N = size(A_adv, 1)
    T = eltype(A_adv)
    y = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    solver = Krylov.CgSolver(N, N, VT)
    solver.x .= zero(T)
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)
    solver_adv = IterativeSolverToolkit(A_adv, P_adv, y, solver, kwargs, "Advection")
    solver_diff = IterativeSolverToolkit(A_diff, P_diff, y, solver, kwargs, "Diffusion")
    return EvolutionToolkit(B_diff, b_diff, solver_adv, solver_diff)
end

function evolve_diffusion!(evolution::EvolutionToolkit, b)
    # calculate rhs vector
    arch = architecture(evolution.B_diff)
    evolution.solver_diff.y .= evolution.B_diff*on_architecture(arch, b.free_values) + evolution.b_diff

    # solve
    iterative_solve!(evolution.solver_diff)

    return evolution
end