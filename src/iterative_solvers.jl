mutable struct IterativeSolverToolkit{A, P, V, S}
    A::A               # LHS matrix
    P::P               # preconditioner for A
    x::V               # solution vector
    y::V               # RHS vector
    is_solved::Bool    # flag to indicate if system has already been solved for this RHS
    solver::S          # iterative solver
    kwargs::Dict       # keyword arguments for iterative solver
    label::String      # label for solver
end

function IterativeSolverToolkit(A, P, y, solver, kwargs, label)
    # x just points to solver.x
    # start with is_solved = false
    return IterativeSolverToolkit(A, P, solver.x, y, false, solver, kwargs, label)
end

function iterative_solve!(solver_tk::IterativeSolverToolkit)
    # unpack
    A = solver_tk.A
    P = solver_tk.P
    y = solver_tk.y
    is_solved = solver_tk.is_solved
    solver = solver_tk.solver
    kwargs = solver_tk.kwargs
    label = solver_tk.label

    if is_solved
        @warn "'$label' system has already been solved for this RHS, skipping solve"
        return solver_tk
    end

    # solve
    Krylov.solve!(solver, A, y, solver.x; M=P, kwargs...)

    # update flag
    solver_tk.is_solved = true

    @debug begin 
        solved = solver.stats.solved
        niter = solver.stats.niter 
        time = solver.stats.timer
        "$label iterative solve: solved=$solved, niter=$niter, time=$time" 
    end

    return solver_tk
end