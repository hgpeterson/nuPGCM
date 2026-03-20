mutable struct IterativeSolverToolkit{A, P, V, S, K}
    A::A           # LHS matrix
    P::P           # preconditioner for A
    x::V           # solution vector
    y::V           # RHS vector
    workspace::S   # Krylov.jl Workspace
    kwargs::K      # keyword arguments for workspace
    label::String  # label for solver
end

function Base.summary(solver::IterativeSolverToolkit)
    t = typeof(solver)
    return "$(parentmodule(t)).$(nameof(t))"
end
function Base.show(io::IO, solver::IterativeSolverToolkit)
    println(io, summary(solver), ":")
    println(io, "├── A: ", summary(solver.A))
    println(io, "├── P: ", summary(solver.P))
    println(io, "├── x: ", summary(solver.x))
    println(io, "├── y: ", summary(solver.y))
    println(io, "├── workspace: ", summary(solver.workspace))
    println(io, "├── kwargs: ", solver.kwargs)
    println(io, "└── label: \"", solver.label, "\"")
end

function IterativeSolverToolkit(A, P, y, workspace, kwargs, label)
    # x just points to workspace.x
    return IterativeSolverToolkit(A, P, workspace.x, y, workspace, kwargs, label)
end

function iterative_solve!(solver_tk::IterativeSolverToolkit)
    # unpack
    A = solver_tk.A
    P = solver_tk.P
    x = solver_tk.x
    y = solver_tk.y
    workspace = solver_tk.workspace
    kwargs = solver_tk.kwargs
    label = solver_tk.label

    # do a direct solve if possible
    if typeof(P) <: Factorization
        t0 = time()
        ldiv!(x, P, y)
        t1 = time()
        @debug @sprintf("Direct %s solve: time=%.3e", label, t1-t0)
        return solver_tk
    end
    if architecture(A) == CPU() && size(A, 1) < 300_000
        t0 = time()
        x .= A\y
        t1 = time()
        @debug @sprintf("Direct %s solve: time=%.3e", label, t1-t0)
        return solver_tk
    end

    # solve
    Krylov.krylov_solve!(workspace, A, y, x; M=P, kwargs...)

    @debug begin 
        solved = workspace.stats.solved
        niter = workspace.stats.niter 
        time = workspace.stats.timer
        @sprintf("%s iterative solve: solved=%s, niter=%d, time=%.3e", label, solved, niter, time) 
    end

    return solver_tk
end