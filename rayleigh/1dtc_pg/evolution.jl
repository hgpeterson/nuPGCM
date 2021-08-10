"""
    matrices = getEvolutionMatrices()

Compute the matrices needed for evolution equation integration.
"""
function getEvolutionMatrices()
    diffMat = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix 
    diffVec = zeros(nẑ)                            # diffusion operator vector 
    bdyFluxMat = Tuple{Int64,Int64,Float64}[]      # flux at boundary matrix

    # interior nodes only for operators
    for j=2:nẑ-1
        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 1)
        κ_ẑ = sum(fd_ẑ.*κ[j-1:j+1])

        # dẑẑ stencil
        fd_ẑẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 2)

        # diffusion term: dẑ(κ(N^2*cos(θ) + dẑ(b))) = dẑ(κ)*N^2*cos(θ) + dẑ(κ)*dẑ(b) + κ*dẑẑ(b)
        push!(diffMat, (j, j-1, (κ_ẑ*fd_ẑ[1] + κ[j]*fd_ẑẑ[1])))
        push!(diffMat, (j, j,   (κ_ẑ*fd_ẑ[2] + κ[j]*fd_ẑẑ[2])))
        push!(diffMat, (j, j+1, (κ_ẑ*fd_ẑ[3] + κ[j]*fd_ẑẑ[3])))
        diffVec[j] = κ_ẑ*N^2*cos(θ)
    end

    # flux at boundaries: bottom
    # dẑ stencil
    fd_ẑ = mkfdstencil(ẑ[1:3], ẑ[1], 1)
    # flux term: dẑ(b) = -N^2*cos(θ)
    push!(bdyFluxMat, (1, 1, fd_ẑ[1]))
    push!(bdyFluxMat, (1, 2, fd_ẑ[2]))
    push!(bdyFluxMat, (1, 3, fd_ẑ[3]))

    # flux at boundaries: top
    # dẑ stencil
    fd_ẑ = mkfdstencil(ẑ[nẑ-2:nẑ], ẑ[nẑ], 1)
    # flux term: dẑ(b) = 0
    push!(bdyFluxMat, (nẑ, nẑ-2, fd_ẑ[1]))
    push!(bdyFluxMat, (nẑ, nẑ-1, fd_ẑ[2]))
    push!(bdyFluxMat, (nẑ, nẑ,   fd_ẑ[3]))

    # Create CSC sparse matrix from matrix elements
    diffMat = sparse((x->x[1]).(diffMat), (x->x[2]).(diffMat), (x->x[3]).(diffMat), nẑ, nẑ)
    bdyFluxMat = sparse((x->x[1]).(bdyFluxMat), (x->x[2]).(bdyFluxMat), (x->x[3]).(bdyFluxMat), nẑ, nẑ)

    return diffMat, diffVec, bdyFluxMat
end

"""
    evolutionLHS = getEvolutionLHS(Δt, diffMat, bdyFluxMat)

Generate the left-hand side matrix for the evolution problem of the form `I - diffmat*Δt`
and the not flux boundary condition applied to the boundaries
"""
function getEvolutionLHS(Δt, diffMat, bdyFluxMat)
    # implicit euler
    evolutionLHS = I - diffMat*Δt 

    # no flux boundaries
    evolutionLHS[1, :] = bdyFluxMat[1, :]
    evolutionLHS[nẑ, :] = bdyFluxMat[nẑ, :]

    return evolutionLHS
end

"""
    b = evolve(tFinal)

Solve equation for `b` for `tFinal` seconds.
"""
function evolve(tFinal)
    # timestep
    nSteps = Int64(ceil(tFinal/Δt))
    nStepsSave = Int64(floor(tSave/Δt))

    # get matrices and vectors
    diffMat, diffVec, bdyFluxMat = getEvolutionMatrices()

    # left-hand side for evolution equation (save LU decomposition for speed)
    evolutionLHS = lu(getEvolutionLHS(Δt, diffMat, bdyFluxMat))

    # initial condition
    t = 0
    b = zeros(nẑ)
    χ, û, v̂, U = invert(b)
    iSave = 0
    saveCheckpoint1DTCPGRayleigh(b, χ, û, v̂, U, t, iSave)
    iSave += 1

    # main loop
    for i=1:nSteps
        t += Δt

        # implicit euler diffusion
        diffRHS = b + diffVec*Δt

        # compute advection RHS
        advRHS = @. -Δt*û*N^2*sin(θ)

        # sum the two
        evolutionRHS = diffRHS + advRHS

        # boundary fluxes
        evolutionRHS[1] = -N^2*cos(θ) 
        #= evolutionRHS[nẑ] = -N^2*cos(θ) =# 
        evolutionRHS[nẑ] = 0

        # solve
        b = evolutionLHS\evolutionRHS

        # invert buoyancy for flow
        χ, û, v̂, U = invert(b)

        if i % nStepsSave == 0
            # log
            println(@sprintf("t = %.2f years (i = %d)", t/secsInYear, i))
            saveCheckpoint1DTCPGRayleigh(b, χ, û, v̂, U, t, iSave)
            iSave += 1
        end
    end

    return b
end