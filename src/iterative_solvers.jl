mutable struct IterativeSolverToolkit{A, P, V, S, K}
    A::A           # LHS matrix
    P::P           # preconditioner for A
    x::V           # solution vector
    y::V           # RHS vector
    solver::S      # iterative solver
    kwargs::K      # keyword arguments for iterative solver
    label::String  # label for solver
end

function IterativeSolverToolkit(A, P, y, solver, kwargs, label)
    # x just points to solver.x
    return IterativeSolverToolkit(A, P, solver.x, y, solver, kwargs, label)
end

function iterative_solve!(solver_tk::IterativeSolverToolkit)
    # unpack
    A = solver_tk.A
    P = solver_tk.P
    x = solver_tk.x
    y = solver_tk.y
    solver = solver_tk.solver
    kwargs = solver_tk.kwargs
    label = solver_tk.label

    # do a direct solve if possible
    if typeof(P) <: Factorization
        ldiv!(x, P, y)
        @debug "Direct $label solve: complete." 
        return solver_tk
    end

    # solve
    Krylov.solve!(solver, A, y, x; M=P, kwargs...)

    @debug begin 
        solved = solver.stats.solved
        niter = solver.stats.niter 
        time = solver.stats.timer
        "$label iterative solve: solved=$solved, niter=$niter, time=$time" 
    end

    return solver_tk
end