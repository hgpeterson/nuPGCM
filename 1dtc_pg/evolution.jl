"""
    matrices = getDiffusionMatrix()

Compute the diffusion matrix needed for evolution equation integration.
"""
function getDiffusionMatrix(z::Array{Float64,1}, κ::Array{Float64,1})
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
    evolutionLHS = getEvolutionLHS(m)

Generate the left-hand side matrix for the evolution problem of the form `I - Δt/2*D`
and the no flux boundary condition applied to the boundaries
"""
function getEvolutionLHS(m::ModelSetup1DPG)
    # implicit euler
    evolutionLHS = I - m.Δt/2*m.D 

    # no flux boundaries
    evolutionLHS[1, :] = m.D[1, :]
    evolutionLHS[m.nz, :] = m.D[m.nz, :]

    return lu(evolutionLHS)
end

"""
    evolve!(m, s, tFinal, tSave; bl=bl)

Solve equation for `b` for `tFinal` seconds.
"""
function evolve!(m::ModelSetup1DPG, s::ModelState1DPG, tFinal::Real, tSave::Real; bl=false)
    # timestep
    nSteps = Int64(tFinal/m.Δt)
    nStepsSave = Int64(tSave/m.Δt)

    # left-hand side for evolution equation
    LHS = getEvolutionLHS(m)

    # initial condition
    iSave = 0
    saveState1DPG(s, iSave)
    iSave += 1

    # main loop
    t = m.Δt
    for i=1:nSteps
        # impose tidally varying U
        m.U[1] = m.Uamp*sin(2*π*t/m.Uper)

        # right-hand side
        RHS = s.b + m.Δt*(1/2*m.D*s.b + m.κ_z*m.N2 - s.u*m.N2*tan(m.θ))

        # reset boundary conditions
        if bl
            RHS[1] = -m.N2/(1 + m.ν[1]/m.κ[1]*m.N2/m.f^2*tan(m.θ)^2)
        else
            RHS[1] = -m.N2
        end
        RHS[m.nz] = 0

        # solve
        s.b[:] = LHS\RHS

        # invert buoyancy for flow
        invert!(m, s; bl=bl)

        if i % nStepsSave == 0
            # log
            println(@sprintf("t = %.2f years (i = %d)", m.Δt*i/secsInYear, i))
            
            # save
            saveState1DPG(s, iSave)

            # plot
            setupFile = string(outFolder, "setup.h5")
            stateFile = @sprintf("%sstate%d.h5", outFolder, iSave)
            imgFile = @sprintf("%sprofiles_%03d.png", outFolder, iSave)
            profilePlot(setupFile, stateFile, imgFile)

            # next
            iSave += 1
        end

        # step
        s.i[1] = i + 1
        t += m.Δt
    end
end
