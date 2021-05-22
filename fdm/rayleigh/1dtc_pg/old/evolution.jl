"""
    matrices = getEvolutionMatrices()

Compute the matrices needed for evolution equation integration.
"""
function getEvolutionMatrices()
    nPts = nx*nz

    umap = reshape(1:nPts, nx, nz)    
    diffMat = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix 
    diffVec = zeros(nPts)                          # diffusion operator vector 
    bdyFluxMat = Tuple{Int64,Int64,Float64}[]      # flux at boundary matrix

    # Main loop, insert stencil in matrices for each node point
    for i=1:nx
        # periodic in x
        iL = mod1(i-1, nx)
        iR = mod1(i+1, nx)

        # interior nodes only for operators
        for j=2:nz-1
            row = umap[i, j] 

            # dẑ stencil
            fd_ẑ = mkfdstencil(ẑẑ[i, j-1:j+1], ẑẑ[i, j], 1)
            κ_ẑ = sum(fd_ẑ.*κ[i, j-1:j+1])

            # dẑẑ stencil
            fd_ẑẑ = mkfdstencil(ẑẑ[i, j-1:j+1], ẑẑ[i, j], 2)

            # diffusion term: dẑ(κ(N^2*cos(θ) + dẑ(b))) = dẑ(κ)*N^2*cos(θ) + dẑ(κ)*dẑ(b) + κ*dẑẑ(b)
            push!(diffMat, (row, umap[i, j-1], (κ_ẑ*fd_ẑ[1] + κ[i, j]*fd_ẑẑ[1])))
            push!(diffMat, (row, umap[i, j],   (κ_ẑ*fd_ẑ[2] + κ[i, j]*fd_ẑẑ[2])))
            push!(diffMat, (row, umap[i, j+1], (κ_ẑ*fd_ẑ[3] + κ[i, j]*fd_ẑẑ[3])))
            diffVec[row] = κ_ẑ*N^2*cosθ[i, j]
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑẑ[i, 1:3], ẑẑ[i, 1], 1)
        # flux term: dẑ(b) = -N^2*cos(θ)
        push!(bdyFluxMat, (row, umap[i, 1], fd_ẑ[1]))
        push!(bdyFluxMat, (row, umap[i, 2], fd_ẑ[2]))
        push!(bdyFluxMat, (row, umap[i, 3], fd_ẑ[3]))

        # flux at boundaries: top
        row = umap[i, nz] 
        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑẑ[i, nz-2:nz], ẑẑ[i, nz], 1)
        # flux term: dẑ(b) = 0
        push!(bdyFluxMat, (row, umap[i, nz-2], fd_ẑ[1]))
        push!(bdyFluxMat, (row, umap[i, nz-1], fd_ẑ[2]))
        push!(bdyFluxMat, (row, umap[i, nz],   fd_ẑ[3]))
    end

    # Create CSC sparse matrix from matrix elements
    diffMat = sparse((x->x[1]).(diffMat), (x->x[2]).(diffMat), (x->x[3]).(diffMat), nPts, nPts)
    bdyFluxMat = sparse((x->x[1]).(bdyFluxMat), (x->x[2]).(bdyFluxMat), (x->x[3]).(bdyFluxMat), nPts, nPts)

    return diffMat, diffVec, bdyFluxMat
end

"""
    evolutionLHS = getEvolutionLHS(Δt, diffMat, bdyFluxMat, bottomBdy, topBdy)

Generate the left-hand side matrix for the evolution problem of the form `I - diffmat*Δt`
and the not flux boundary condition applied to the boundaries
"""
function getEvolutionLHS(Δt, diffMat, bdyFluxMat, bottomBdy, topBdy)
    # implicit euler
    evolutionLHS = I - diffMat*Δt 

    # no flux boundaries
    evolutionLHS[bottomBdy, :] = bdyFluxMat[bottomBdy, :]
    evolutionLHS[topBdy, :] = bdyFluxMat[topBdy, :]

    return evolutionLHS
end

"""
    b = evolve(nSteps)

Solve equation for `b` for `nSteps` time steps.
"""
function evolve(nSteps)
    # grid points
    nPts = nx*nz

    # timestep
    Δt = 10*86400
    nStepsInvert = 1
    nStepsPlot = 10
    nStepsSave = 100
    adaptiveTimestep = false

    # for flattening for matrix mult
    umap = reshape(1:nPts, nx, nz)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nz]

    # get matrices and vectors
    diffMat, diffVec, bdyFluxMat = getEvolutionMatrices()

    # left-hand side for evolution equation (save LU decomposition for speed)
    evolutionLHS = lu(getEvolutionLHS(Δt, diffMat, bdyFluxMat, bottomBdy, topBdy))

    # left-hand side for inversion equations
    inversionLHS = lu(getInversionLHS())

    # initial condition
    t = 0
    b = zeros(nx, nz)
    #= # load data =#
    #= file = h5open("b.h5", "r") =#
    #= b = read(file, "b") =#
    #= t = read(file, "t") =#
    #= close(file) =#

    # invert initial condition
    chi, û, v = invert(b, inversionLHS)

    # flatten for matrix mult
    bVec = reshape(b, nPts, 1)
    ûVec = reshape(û, nPts, 1)
    vVec = reshape(v, nPts, 1)
    sinθVec = reshape(sinθ, nPts, 1)
    
    # plot initial state of all zeros and no flow
    iImg = 0
    plotCurrentState(t, chi, û, v, b, iImg)

    # main loop
    for i=1:nSteps
        t += Δt
        tDays = t/86400

        # implicit euler diffusion
        diffRHS = bVec + diffVec*Δt

        # compute advection RHS
        advRHS = @. -ûVec*N^2*sinθVec

        # sum the two
        evolutionRHS = diffRHS + advRHS

        # boundary fluxes
        evolutionRHS[bottomBdy] .= -N^2*cosθ[:, 1] 
        evolutionRHS[topBdy] .= 0

        # solve
        bVec = evolutionLHS\evolutionRHS

        # log
        println(@sprintf("t = %.2f days (i = %d)", tDays, i))
        if i % nStepsInvert == 0
            # reshape
            b = reshape(bVec, nx, nz)

            # invert buoyancy for flow
            chi, û, v = invert(b, inversionLHS)
            ûVec = reshape(û, nPts, 1)
            vVec = reshape(v, nPts, 1)
        end
        if i % nStepsPlot == 0
            # plot flow
            iImg += 1
            plotCurrentState(t, chi, û, v, b, iImg)
        end
        if i % nStepsSave == 0
            # save data
            savefile = @sprintf("b%d.h5", tDays)
            println("saving to ", savefile)
            file = h5open(savefile, "w")
            write(file, "b", b)
            write(file, "t", t)
            close(file)
        end
    end

    b = reshape(bVec, nx, nz)

    return b
end
