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

"""
    update_A!(solver_A, A_new::SparseMatrixCSC, gpu_perm)

Overwrite the values of `solver_A` with those of `A_new`, which has the same
sparsity pattern. `gpu_perm` caches the CSC → CSR permutation of the nonzeros on
the GPU and is unused on the CPU.
"""
function update_A!(solver_A::SparseMatrixCSC, A_new::SparseMatrixCSC, gpu_perm)
    solver_A.nzval .= A_new.nzval
    return solver_A
end

"""
    position = nonzero_positions(A)

Position in `nonzeros(A)` of each entry of `nonzeros(on_architecture(CPU(), A))`.
"""
nonzero_positions(A::SparseMatrixCSC) = eachindex(nonzeros(A))

"""
    update_P!(P, A_cpu, arch)

Recompute the preconditioner `P` in place for the matrix `A_cpu`: the Jacobi
diagonal `1 ./ diag(A_cpu)` for a `Diagonal`, the LU factors for a `Factorization`.
"""
function update_P!(P::Diagonal, A_cpu::SparseMatrixCSC, arch)
    P.diag .= on_architecture(arch, Vector(1 ./ diag(A_cpu)))
    return P
end
function update_P!(P::Factorization, A_cpu::SparseMatrixCSC, arch)
    lu!(P, A_cpu)
    return P
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

    # FGMRES preconditions on the right, so that its residual is ‖y - Ax‖
    preconditioner = workspace isa Krylov.FgmresWorkspace ? (; N=P) : (; M=P)
    Krylov.krylov_solve!(workspace, A, y, x; preconditioner..., kwargs...)

    if !workspace.stats.solved
        r = workspace.stats.residuals
        @warn @sprintf("%s iterative solve did not converge: niter=%d, resid[1]=%.3e, resid[end]=%.3e",
                       label, workspace.stats.niter,
                       isempty(r) ? NaN : r[1], isempty(r) ? NaN : r[end])
    end

    @debug begin 
        solved = workspace.stats.solved
        niter = workspace.stats.niter 
        time = workspace.stats.timer
        residual₁ = workspace.stats.residuals[1]
        residualₑ = workspace.stats.residuals[end]
        @sprintf("%s iterative solve: solved=%s, niter=%d, resid[1]=%.3e, resid[end]=%.3e, time=%.3e", label, solved, niter, residual₁, residualₑ, time) 
    end

    return solver_tk
end