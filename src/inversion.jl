struct InversionToolkit{B}
    B::B  # RHS matrix
    solver::IterativeSolverToolkit
end

function InversionToolkit(A, P, B; atol=1e-6, rtol=1e-6, itmax=0, memory=20, history=true, verbose=false)
    arch = architecture(A)
    N = size(A, 1)
    T = eltype(A)
    y = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    solver = Krylov.GmresSolver(N, N, memory, VT)
    solver.x .= zero(T)
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int)
    solver_tk = IterativeSolverToolkit(A, P, y, solver, kwargs, "Inversion") 
    return InversionToolkit(B, solver_tk)
end

function invert!(inversion::InversionToolkit, b)
    # unpack
    solver = inversion.solver
    y = solver.y
    arch = architecture(y)
    B = inversion.B

    # calculate rhs vector
    y .= B*on_architecture(arch, b.free_values)
    solver.is_solved = false

    # solve
    iterative_solve!(solver)

    return inversion
end