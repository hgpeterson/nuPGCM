struct InversionToolkit{A, P, V, S}
    lhs_matrix::A 
    preconditioner::P 
    rhs_matrix::A 
    rhs_vector::V
    solver::S
    atol::Real
    rtol::Real
    itmax::Int
    memory::Int
    history::Bool
    verbose::Bool
end

function InversionToolkit(lhs_matrix, preconditioner, rhs_matrix; 
                          atol=1e-6, rtol=1e-6, itmax=0, memory=20, history=true, verbose=false)
    arch = architecture(lhs_matrix)
    N = size(lhs_matrix, 1)
    T = eltype(lhs_matrix)
    rhs_vector = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    solver = Krylov.GmresSolver(N, N, memory, VT)
    solver.x .= zero(T)
    return InversionToolkit(lhs_matrix, preconditioner, rhs_matrix, rhs_vector, solver, 
                            atol, rtol, itmax, memory, history, verbose)
end

function invert!(inversion::InversionToolkit, b)
    # unpack
    solver = inversion.solver
    A = inversion.lhs_matrix
    P = inversion.preconditioner
    y = inversion.rhs_vector
    atol = inversion.atol
    rtol = inversion.rtol
    itmax = inversion.itmax
    history = inversion.history
    verbose = inversion.verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int

    # calculate rhs vector
    calculate_inversion_rhs_vector!(inversion, b)

    # solve
    Krylov.solve!(solver, A, y, solver.x;  
                  atol, rtol, itmax, history, verbose,
                  M=P, restart=true)

    @debug begin 
    solved = solver.stats.solved
    niter = solver.stats.niter 
    time = solver.stats.timer
    "Inversion iterative solve: solved=$solved, niter=$niter, time=$time" 
    end

    return inversion
end

function calculate_inversion_rhs_vector!(inversion::InversionToolkit, b)
    arch = architecture(inversion.rhs_vector)
    inversion.rhs_vector .= inversion.rhs_matrix*on_architecture(arch, b.free_values)
    return inversion
end