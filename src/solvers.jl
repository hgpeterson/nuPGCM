struct IterativeSolver{S, A, P, B, T}
    state::S 
    A::A
    P::P
    B::B
    atol::T
    rtol::T
    itmax::Int
end

function generate_inversion_solver(arch, A, P, B; atol=1e-6, rtol=1e-6, itmax=0, memory=20)
    N = size(A, 1)
    T = eltype(A)
    VT = vector_type(arch, T)
    state = Krylov.GmresSolver(N, N, memory, VT)
    return IterativeSolver(state, A, P, B, atol, rtol, itmax)
end

function solve_inversion!(solver::IterativeSolver, y; verbose=0)
    arch = architecture(solver.state.x)
    y_arch = on_architecture(arch, y.free_values)
    rhs = solver.B*y_arch
    Krylov.solve!(solver.state, solver.A, rhs, solver.state.x, M=solver.P,
                  atol=solver.atol, rtol=solver.rtol, verbose=verbose, itmax=solver.itmax, 
                  restart=true, history=true)
    @debug "Inversion GMRES solve" solved=solver.state.stats.solved niter=solver.state.stats.niter time=solver.state.stats.timer
    return solver
end

function generate_evolution_solver(arch, A, P, B; atol=1e-6, rtol=1e-6, itmax=0)
    N = size(A, 1)
    T = eltype(A)
    VT = vector_type(arch, T)
    state = Krylov.CgSolver(N, N, VT)
    return IterativeSolver(state, A, P, B, atol, rtol, itmax)
end

function solve_evolution!(solver::IterativeSolver, y; verbose=0)
    arch = architecture(solver.state.x)
    y_arch = on_architecture(arch, y)
    rhs = solver.B*y_arch
    Krylov.solve!(solver.state, solver.A, rhs, solver.state.x, M=solver.P,
                  atol=solver.atol, rtol=solver.rtol, verbose=verbose, itmax=solver.itmax, 
                  history=true)
    @debug "Evolution CG solve" solved=solver.state.stats.solved niter=solver.state.stats.niter time=solver.state.stats.timer
    return solver
end

function vector_type(arch::CPU, T)
    return Vector{T}
end
function vector_type(arch::GPU, T)
    return CuVector{T}
end

# if typeof(dim) == TwoD && typeof(arch) == CPU
#     @time "lu(LHS_inversion)" P_inversion = lu(LHS_inversion)
#     ldiv_P_inversion = true
# else
#     P_inversion = Diagonal(on_architecture(arch, 1/h^dim.n*ones(N)))
#     # P_inversion = I
#     ldiv_P_inversion = false
# end

# function invert!(arch::AbstractArchitecture, solver, b)
#     b_arch = on_architecture(arch, b.free_values)
#     if typeof(arch) == GPU
#         RHS = [CUDA.zeros(nx); CUDA.zeros(ny); RHS_inversion*b_arch; CUDA.zeros(np-1)]
#     else
#         RHS = [zeros(nx); zeros(ny); RHS_inversion*b_arch; zeros(np-1)]
#     end
#     Krylov.solve!(solver, LHS_inversion, RHS, solver.x, M=P_inversion, ldiv=ldiv_P_inversion,
#                   atol=tol, rtol=tol, verbose=1, itmax=itmax, restart=true,
#                   history=true)
#     @printf("inversion GMRES solve: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
#     return solver
# end

# if typeof(dim) == TwoD && typeof(arch) == CPU
#     @time "lu(LHS_diff)" P_diff = lu(LHS_diff)
#     @time "lu(LHS_adv)"  P_adv  = lu(LHS_adv)
#     ldiv_P_diff = true
#     ldiv_P_adv  = true
# else
#     P_diff = Diagonal(on_architecture(arch, Vector(1 ./ diag(LHS_diff))))
#     P_adv  = Diagonal(on_architecture(arch, Vector(1 ./ diag(LHS_adv))))
#     ldiv_P_diff = false
#     ldiv_P_adv  = false
# end

# solver_evolution = CgSolver(nb, nb, VT)
# solver_evolution.x .= on_architecture(arch, copy(b.free_values)[perm_b])

# b_half = interpolate_everywhere(0, B)
# function evolve_adv!(arch::AbstractArchitecture, solver_inversion, solver_evolution, ux, uy, uz, p, b)
#     # half step
#     l_half(d) = ∫( b*d - Δt/2*(ux*∂x(b) + uy*∂y(b) + uz*(N² + ∂z(b)))*d )dΩ
#     @time "build RHS_evolution 1" RHS = on_architecture(arch, assemble_vector(l_half, D)[perm_b])
#     Krylov.solve!(solver_evolution, LHS_adv, RHS, solver_evolution.x, M=P_adv, ldiv=ldiv_P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)
#     @printf("advection CG solve 1: solved=%s, niter=%d, time=%f\n", solver_evolution.stats.solved, solver_evolution.stats.niter, solver_evolution.stats.timer)

#     # u, v, w, p, b at half step
#     update_b!(b_half, solver_evolution)
#     solver_inversion = invert!(arch, solver_inversion, b_half)
#     ux, uy, uz, p = update_u_p!(ux, uy, uz, p, solver_inversion)

#     # full step
#     l_full(d) = ∫( b*d - Δt*(ux*∂x(b_half) + uy*∂y(b_half) + uz*(N² + ∂z(b_half)))*d )dΩ
#     @time "build RHS_evolution 2" RHS = on_architecture(arch, assemble_vector(l_full, D)[perm_b])
#     Krylov.solve!(solver_evolution, LHS_adv, RHS, solver_evolution.x, M=P_adv, ldiv=ldiv_P_adv, atol=tol, rtol=tol, verbose=0, itmax=itmax)
#     @printf("advection CG solve 2: solved=%s, niter=%d, time=%f\n", solver_evolution.stats.solved, solver_evolution.stats.niter, solver_evolution.stats.timer)

#     return solver_inversion, solver_evolution
# end
# function evolve_diff!(arch::AbstractArchitecture, solver, b)
#     b_arch= on_architecture(arch, b.free_values)
#     RHS = RHS_diff*b_arch + rhs_diff
#     Krylov.solve!(solver, LHS_diff, RHS, solver.x, M=P_diff, ldiv=ldiv_P_diff, atol=tol, rtol=tol, verbose=0, itmax=itmax)
#     @printf("diffusion CG solve: solved=%s, niter=%d, time=%f\n", solver.stats.solved, solver.stats.niter, solver.stats.timer)
#     return solver
# end