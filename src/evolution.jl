struct EvolutionToolkit{A, P, V, S}
    lhs_adv_matrix::A 
    preconditioner_adv::P 
    lhs_diff_matrix::A 
    preconditioner_diff::P 
    rhs_diff_matrix::A 
    rhs_diff_vector::V
    rhs_vector::V
    solver::S
    atol::Real
    rtol::Real
    itmax::Int
    history::Bool
    verbose::Bool
end

function EvolutionToolkit(lhs_adv_matrix, preconditioner_adv, lhs_diff_matrix, preconditioner_diff, rhs_diff_matrix, rhs_diff_vector;
                          atol=1e-6, rtol=1e-6, itmax=0, history=true, verbose=false)
    arch = architecture(lhs_adv_matrix)
    N = size(lhs_adv_matrix, 1)
    T = eltype(lhs_adv_matrix)
    rhs_vector = on_architecture(arch, zeros(T, N))
    VT = vector_type(arch, T)
    solver = Krylov.CgSolver(N, N, VT)
    solver.x .= zero(T)
    return EvolutionToolkit(lhs_adv_matrix, preconditioner_adv, lhs_diff_matrix, preconditioner_diff, rhs_diff_matrix, rhs_diff_vector, rhs_vector, solver,
                            atol, rtol, itmax, history, verbose)
end

function calculate_diffusion_rhs_vector!(evolution::EvolutionToolkit, b)
    arch = architecture(evolution.rhs_vector)
    evolution.rhs_vector .= evolution.rhs_diff_matrix*on_architecture(arch, b.free_values) + evolution.rhs_diff_vector
    return evolution
end

function evolve_diffusion!(evolution::EvolutionToolkit, b)
    # calculate rhs vector, then invert
    calculate_diffusion_rhs_vector!(evolution, b)
    return evolve_diffusion!(evolution)
end
function evolve_diffusion!(evolution::EvolutionToolkit)
    # unpack
    solver = evolution.solver
    A = evolution.lhs_diff_matrix
    P = evolution.preconditioner_diff
    y = evolution.rhs_vector
    atol = evolution.atol
    rtol = evolution.rtol
    itmax = evolution.itmax
    history = evolution.history
    verbose = evolution.verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int

    # solve
    Krylov.solve!(solver, A, y, solver.x;  
                  atol, rtol, itmax, history, verbose,
                  M=P)

    @debug begin 
    solved = solver.stats.solved
    niter = solver.stats.niter 
    time = solver.stats.timer
    "Diffusion iterative solve: solved=$solved, niter=$niter, time=$time" 
    end

    return evolution
end

# function calculate_advection_rhs_vector!(evolution::EvolutionToolkit, mesh::Mesh, params::Parameters, u, v, w, b, Δt)
#     # unpack
#     arch = architecture(evolution.rhs_vector)
#     p_b = mesh.dofs.p_b
#     B_test = mesh.spaces.B_test
#     dΩ = mesh.dΩ
#     N² = params.N²

#     # half step
#     l_half(d) = ∫( b*d - Δt/2*(u*∂x(b) + v*∂y(b) + w*(N² + ∂z(b)))*d )dΩ
#     evolution.rhs_vector .= on_architecture(arch, assemble_vector(l_half, B_test)[p_b])
#     return evolution
# end

# function evolve_advection!(evolution::EvolutionToolkit, mesh::Mesh, params::Parameters, u, v, w, b, Δt)
#     # calculate rhs vector, then invert
#     calculate_advection_rhs_vector!(evolution, mesh, params, u, v, w, b, Δt)
#     return evolve_advection!(evolution)
# end
# function evolve_advection!(evolution::EvolutionToolkit)
#     # unpack
#     solver = evolution.solver
#     A = evolution.lhs_adv_matrix
#     P = evolution.preconditioner_adv
#     y = evolution.rhs_vector
#     atol = evolution.atol
#     rtol = evolution.rtol
#     itmax = evolution.itmax
#     history = evolution.history
#     verbose = evolution.verbose ? 1 : 0 # I like to have verbose be a Bool but Krylov expects an Int

#     # solve
#     Krylov.solve!(solver, A, y, solver.x;  
#                   atol, rtol, itmax, history, verbose,
#                   M=P)

#     @debug begin 
#     solved = solver.stats.solved
#     niter = solver.stats.niter 
#     time = solver.stats.timer
#     "Advection iterative solve: solved=$solved, niter=$niter, time=$time" 
#     end

#     return evolution
# end