"""
    evolution_LHS = get_evolution_LHS(m)

Generate the left-hand side matrix for the evolution problem of the form `I - Δt/2*D`
and the no flux boundary condition applied to the boundaries
"""
function get_evolution_LHS(m::ModelSetup1D)
    # implicit euler
    evolution_LHS = I - m.Δt/2*m.D 

    # no flux boundaries
    evolution_LHS[1, :] = m.D[1, :]
    evolution_LHS[m.nz, :] = m.D[m.nz, :]

    return lu(evolution_LHS)
end

"""
    evolve!(m, s, t_final, t_save)

Solve equation for `b` for `t_final` seconds.
"""
function evolve!(m::ModelSetup1D, s::ModelState1D, t_final, t_save)
    # timestep
    n_steps = Int64(t_final/m.Δt)
    n_steps_save = Int64(t_save/m.Δt)

    # left-hand side for evolution equation
    LHS = get_evolution_LHS(m)

    # initial condition
    i_save = 0
    save_state(s, i_save)
    i_save += 1

    # main loop
    t = m.Δt
    for i=1:n_steps
        # right-hand side
        RHS = s.b + m.Δt*(1/2*m.D*s.b + m.κ_z*m.N2 - s.u*m.N2*tan(m.θ))

        # reset boundary conditions
        if m.bl
            RHS[1] = (m.U[1]*m.N2*tan(m.θ)/m.κ[1] - m.N2)/(1 + m.ν[1]/m.κ[1]*m.N2/m.f^2*tan(m.θ)^2)
        else
            RHS[1] = -m.N2
        end
        RHS[m.nz] = 0

        # solve
        s.b[:] = LHS\RHS

        # invert buoyancy for flow
        invert!(m, s)
        if any(isnan.(s.χ))
            error("Broke on iteration $i.")
        end

        if i % n_steps_save == 0
            # log
            println(@sprintf("t = %.2f years (i = %d)", m.Δt*i/secs_in_year, i))
            
            # save
            save_state(s, i_save)

            # next
            i_save += 1
        end

        # step
        s.i[1] = i + 1
        t += m.Δt
    end
end