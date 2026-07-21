# Immutable: `A`, `P`, `x`, and `y` are all containers updated in place
# (`update_A!` for A, `P.diag .= …` for the Diagonal preconditioner, and
# broadcast into x/y), so no field is ever reassigned after construction.
struct IterativeSolverToolkit{A, P, V, S, K}
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

Refresh `solver_A`'s values in place from `A_new`, which must share `solver_A`'s
sparsity pattern (only nonzero *values* differ). `gpu_perm` is an
architecture-specific scratch `Ref` (see [`InversionLHSCache`](@ref)); the CPU
method below ignores it, the CUDA extension's method uses it to cache the
CSC→CSR nonzero-order permutation (`CuSparseMatrixCSR` stores nonzeros in a
different order than `SparseMatrixCSC`, so a raw `nzval` copy would silently
scramble the matrix).

No new sparse matrix is allocated, unlike `solver.A = on_architecture(arch, A_new)`.
"""
function update_A!(solver_A::SparseMatrixCSC, A_new::SparseMatrixCSC, gpu_perm)
    solver_A.nzval .= A_new.nzval
    return solver_A
end

"""
    update_P!(P, A_cpu, arch)

Refresh the preconditioner `P` in place from the (CPU) LHS `A_cpu`, so the
solver's `P` field never needs reassigning. A `Diagonal` gets its wrapped
vector overwritten with `1 ./ diag(A_cpu)` (on `arch`); a sparse UMFPACK
`Factorization` is refactorized in place via `lu!` (reusing its symbolic
factorization). Both cases require `A_cpu`'s pattern to be unchanged.
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