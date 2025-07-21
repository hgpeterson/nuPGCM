struct InversionToolkit{B, V, S <: IterativeSolverToolkit}
    B::B      # RHS matrix
    b::V      # RHS vector
    solver::S # iterative solver toolkit
end

function InversionToolkit(A, P, B, b; atol=1e-6, rtol=1e-6, itmax=0, memory=20, history=true, verbose=false, restart=true)
    arch = architecture(A)
    N = size(A, 1)
    T = eltype(A)
    y = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    solver = Krylov.GmresSolver(N, N, memory, VT)
    solver.x .= zero(T)
    verbose_int = verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int
    kwargs = Dict(:atol=>atol, :rtol=>rtol, :itmax=>itmax, :history=>history, :verbose=>verbose_int, :restart=>restart)
    solver_tk = IterativeSolverToolkit(A, P, y, solver, kwargs, "Inversion") 
    return InversionToolkit(B, b, solver_tk)
end

function invert!(inversion::InversionToolkit, b)
    # calculate rhs vector
    arch = architecture(inversion.B)
    inversion.solver.y .= inversion.B*on_architecture(arch, b.free_values) .+ inversion.b

    # solve
    iterative_solve!(inversion.solver)

    return inversion
end