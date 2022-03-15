"""
    matrices = get_diffusion_matrix()

Compute the diffusion matrix needed for evolution equation integration.
"""
function get_diffusion_matrix(z::Array{Float64,1}, κ::Array{Float64,1})
    nz = size(z, 1)
    D = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix 

    # interior nodes 
    for j=2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        κ_z = sum(fd_z.*κ[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # diffusion term: dz(κ(N^2*cos(θ) + dz(b))) = dz(κ)*N^2*cos(θ) + dz(κ)*dz(b) + κ*dzz(b)
        push!(D, (j, j-1, (κ_z*fd_z[1] + κ[j]*fd_zz[1])))
        push!(D, (j, j,   (κ_z*fd_z[2] + κ[j]*fd_zz[2])))
        push!(D, (j, j+1, (κ_z*fd_z[3] + κ[j]*fd_zz[3])))
    end

    # flux at boundaries: bottom
    # dz stencil
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    # flux term: dz(b) = -N^2*cos(θ)
    push!(D, (1, 1, fd_z[1]))
    push!(D, (1, 2, fd_z[2]))
    push!(D, (1, 3, fd_z[3]))

    # flux at boundaries: top
    # dz stencil
    fd_z = mkfdstencil(z[nz-2:nz], z[nz], 1)
    # flux term: dz(b) = 0
    push!(D, (nz, nz-2, fd_z[1]))
    push!(D, (nz, nz-1, fd_z[2]))
    push!(D, (nz, nz,   fd_z[3]))

    # Create CSC sparse matrix from matrix elements
    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), nz, nz)

    return D
end

"""
    evolution_LHS = get_evolution_LHS(m)

Generate the left-hand side matrix for the evolution problem of the form `I - Δt/2*D`
and the no flux boundary condition applied to the boundaries
"""
function get_evolution_LHS(m::ModelSetup1DPG)
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
function evolve!(m::ModelSetup1DPG, s::ModelState1DPG, t_final::Real, t_save::Real)
    # timestep
    n_steps = Int64(t_final/m.Δt)
    n_steps_save = Int64(t_save/m.Δt)

    # left-hand side for evolution equation
    LHS = get_evolution_LHS(m)

    # initial condition
    i_save = 0
    save_state_1DPG(s, i_save)
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

        if i % n_steps_save == 0
            # log
            println(@sprintf("t = %.2f years (i = %d)", m.Δt*i/secs_in_year, i))
            
            # save
            save_state_1DPG(s, i_save)

            # next
            i_save += 1
        end

        # step
        s.i[1] = i + 1
        t += m.Δt
    end
end
