################################################################################
# PG evolution functions
################################################################################

"""
    LHS = get_evolution_LHS(m, a)

Generate the left-hand side matrix for the evolution problem with flux boundary conditions on the boundaries.
"""
function get_evolution_LHS(m::ModelSetup2D, a)
    # bottom and top boundaries in 1D
    umap = reshape(1:m.nξ*m.nσ, m.nξ, m.nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, m.nσ]

    # LHS matrix
    LHS = I - a*m.Δt*m.D 

    # no flux boundaries
    LHS[bottomBdy, :] = m.D[bottomBdy, :]
    # LHS[topBdy, :] = m.D[topBdy, :]
    LHS[topBdy, :] .= 0
    for i ∈ topBdy
        LHS[i, i] = 1
    end

    return lu(LHS)
end

"""
    reset_BCs!(m, s, RHS)

Modify the right-hand side vector `RHS` to include boundary conditions at the top and bottom.
"""
function reset_BCs!(m::ModelSetup2D, s::ModelState2D, RHS)
    # boundary fluxes: dσ(b)/H at σ = -1, 0
    if m.bl
        RHS[:, 1] = s.χ[:, 1].*∂ξ(m, s.b[:, 1])./m.κ[:, 1]
        # RHS[:, m.nσ] = s.χ[:, m.nσ].*∂ξ(m, s.b[:, m.nσ])./m.κ[:, m.nσ] .+ m.N2[:, m.nσ]
        RHS[:, m.nσ] .= 0
    else
        RHS[:, 1] .= 0
        # RHS[:, m.nσ] .= m.N2[:, m.nσ]
        RHS[:, m.nσ] .= 0
    end
end

"""
    db = advection_RHS(m::ModelSetup2D, uξ, uσ, b)

Return db = -uξ*∂ξ(b) - uσ*∂σ(b) for the advection term in the evolution equation.
"""
function advection_RHS(m::ModelSetup2D, uξ, uσ, b)
    return -uξ.*∂ξ(m, b) .- uσ.*∂σ(m, b)
end

"""
    evolve!(m, s, t_final, t_plot, t_save)

Solve evoluion equation for `b` and update model state.
"""
function evolve!(m::ModelSetup2D, s::ModelState2D, t_final, t_plot, t_save)
    # timestep
    n_steps      = Int64(round(t_final/m.Δt))
    n_steps_plot = Int64(round(t_plot/m.Δt))
    n_steps_save = Int64(round(t_save/m.Δt))

    # save initial state
    i_save = 0
    save_state(s, i_save)
    i_save += 1
    
    # plot initial state
    i_img = 0
    plot_state(m, s, i_img)
    i_img += 1

    # store previous buoyancy field for timestepping scheme
    b_prev = zeros(m.nξ, m.nσ)

    # get LHS matrix for CNAB
    LHS = get_evolution_LHS(m, 1/2)

    # if you want to check CFL
    dξ = m.L/m.nξ
    dσ = zeros(m.nξ, m.nσ)
    σσ = repeat(m.σ', m.nξ, 1)
    dσ[:, 1:end-1] = σσ[:, 2:end] - σσ[:, 1:end-1]
    dσ[:, end] = dσ[:, end-1]

    # main loop
    t = 0
    uξ_prev = zeros(size(s.uξ))
    uσ_prev = zeros(size(s.uσ))
    b_prev  = zeros(size(s.b))
    RHS     = zeros(size(s.b))
    for i=1:n_steps
        # store previoius step for next time
        uξ_prev[:, :] = s.uξ[:, :]
        uσ_prev[:, :] = s.uσ[:, :]
        b_prev[:, :]  = s.b[:, :]

        # form RHS = b + Δt*(advection + diffusion)
        RHS[:, :] = s.b[:, :]
        # advection
        if i == 1
            # first step: CNAB1
            RHS += m.Δt*advection_RHS(m, s.uξ, s.uσ, s.b)
        else
            # other steps: CNAB2
            RHS += m.Δt*(3/2*advection_RHS(m, s.uξ, s.uσ, s.b) - 1/2*advection_RHS(m, uξ_prev, uσ_prev, b_prev))
        end
        # diffusion
        RHS += m.Δt/2*reshape(m.D*s.b[:], m.nξ, m.nσ)

        # modify RHS to implement boundary conditions
        reset_BCs!(m, s, RHS)

        # solve
        s.b[:, :] = reshape(LHS\RHS[:], m.nξ, m.nσ) 

        # next step
        s.i[1] = i + 1
        t += m.Δt

        # invert buoyancy for flow and save to state
        invert!(m, s)

        # log
        if i % 10 == 0
            if m.no_net_transport
                println(@sprintf("t = %2.2f yr | i = %d | χₘₐₓ = %.2e m2 s-1", t/secs_in_year, i, maximum(abs.(s.χ))))
            else
                println(@sprintf("t = %2.2f yr | i = %d | U = %.2e m2 s-1", t/secs_in_year, i, s.χ[1, end]))
            end

            # # CFL stuff
            # uξCFL = minimum(abs.(dξ./s.uξ)) 
            # uσCFL = minimum(abs.(dσ./s.uσ)) 
            # println(@sprintf("CFL: uξ=%.2f days, uσ=%.2f days", uξCFL/secs_in_day, uσCFL/secs_in_day)) 
        end
        
        if i % n_steps_plot == 0
            # plot flow
            plot_state(m, s, i_img)
            i_img += 1
        end
        if i % n_steps_save == 0
            save_state(s, i_save)
            i_save += 1
        end
    end
end
