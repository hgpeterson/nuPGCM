"""
    matrices = getEvolutionMatrices()

Compute the matrices needed for evolution equation integration.
"""
function getEvolutionMatrices()
    nPts = nξ*nσ

    umap = reshape(1:nPts, nξ, nσ)    
    diffMat = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix 
    diffVec = zeros(nPts)                          # diffusion operator vector 
    bdyFluxMat = Tuple{Int64,Int64,Float64}[]      # flux at boundary matrix
    ξDerivativeMat = Tuple{Int64,Int64,Float64}[]  # advection operator matrix (ξ)
    σDerivativeMat = Tuple{Int64,Int64,Float64}[]  # advection operator matrix (σ)

    # Main loop, insert stencil in matrices for each node point
    for i=1:nξ
        # periodic in ξ
        iL = mod1(i-1, nξ)
        iR = mod1(i+1, nξ)

        # interior nodes only for operators
        for j=2:nσ-1
            row = umap[i, j] 

            # dσ stencil
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            κ_σ = sum(fd_σ.*κ[i, j-1:j+1])

            # dσσ stencil
            fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

            # diffusion term: dσ(κ(N^2 + dσ(b)/H))/H = N^2*dσ(κ)/H + dσ(κ)*dσ(b)/H^2 + κ*dσσ(b)/H^2
            push!(diffMat, (row, umap[i, j-1], (κ_σ*fd_σ[1] + κ[i, j]*fd_σσ[1])/H(ξ[i])^2))
            push!(diffMat, (row, umap[i, j],   (κ_σ*fd_σ[2] + κ[i, j]*fd_σσ[2])/H(ξ[i])^2))
            push!(diffMat, (row, umap[i, j+1], (κ_σ*fd_σ[3] + κ[i, j]*fd_σσ[3])/H(ξ[i])^2))
            diffVec[row] = N^2*κ_σ/H(ξ[i])

            # ξ advection term: dξ()
            push!(ξDerivativeMat, (row, umap[iR, j],  1.0/(2*dξ)))
            push!(ξDerivativeMat, (row, umap[iL, j], -1.0/(2*dξ)))

            # σ advection term: dσ()
            push!(σDerivativeMat, (row, umap[i, j-1], fd_σ[1]))
            push!(σDerivativeMat, (row, umap[i, j],   fd_σ[2]))
            push!(σDerivativeMat, (row, umap[i, j+1], fd_σ[3]))
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
        # flux term: dσ(b)/H = ...
        push!(bdyFluxMat, (row, umap[i, 1], fd_σ[1]/H(ξ[i])))
        push!(bdyFluxMat, (row, umap[i, 2], fd_σ[2]/H(ξ[i])))
        push!(bdyFluxMat, (row, umap[i, 3], fd_σ[3]/H(ξ[i])))

        # flux at boundaries: top
        row = umap[i, nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        # flux term: dσ(b)/H = ...
        push!(bdyFluxMat, (row, umap[i, nσ-2], fd_σ[1]/H(ξ[i])))
        push!(bdyFluxMat, (row, umap[i, nσ-1], fd_σ[2]/H(ξ[i])))
        push!(bdyFluxMat, (row, umap[i, nσ],   fd_σ[3]/H(ξ[i])))
    end

    # Create CSC sparse matrix from matrix elements
    diffMat = sparse((x->x[1]).(diffMat), (x->x[2]).(diffMat), (x->x[3]).(diffMat), nPts, nPts)
    bdyFluxMat = sparse((x->x[1]).(bdyFluxMat), (x->x[2]).(bdyFluxMat), (x->x[3]).(bdyFluxMat), nPts, nPts)
    ξDerivativeMat = sparse((x->x[1]).(ξDerivativeMat), (x->x[2]).(ξDerivativeMat), (x->x[3]).(ξDerivativeMat), nPts, nPts)
    σDerivativeMat = sparse((x->x[1]).(σDerivativeMat), (x->x[2]).(σDerivativeMat), (x->x[3]).(σDerivativeMat), nPts, nPts)

    return diffMat, diffVec, bdyFluxMat, ξDerivativeMat, σDerivativeMat
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
    b = evolve(tFinal; bl=false)

Solve full nonlinear equation for `b` for `tFinal` seconds.
If `bl` set to `true`, use boundary layer theory inversion and boundary conditions. 
"""
function evolve(tFinal; bl=false)
    if bl
        inversion = invertBL
    else
        inversion = invert
    end

    # grid points
    nPts = nξ*nσ

    # timestep
    nSteps = Int64(tFinal/Δt)
    nStepsPlot = Int64(tPlot/Δt)
    nStepsSave = Int64(tSave/Δt)

    # for flattening for matrix mult
    umap = reshape(1:nPts, nξ, nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nσ]

    # get matrices and vectors
    diffMat, diffVec, bdyFluxMat, ξDerivativeMat, σDerivativeMat = getEvolutionMatrices()

    # left-hand side for evolution equation (save LU decomposition for speed)
    evolutionLHS = lu(getEvolutionLHS(Δt, diffMat, bdyFluxMat, bottomBdy, topBdy))

    # vectors of H, Hx, and σ values for the N^*w term
    HVec = reshape(H.(x), nPts, 1)
    HxVec = reshape(Hx.(x), nPts, 1)
    σσVec = reshape(σσ, nPts, 1)

    # initial condition
    t = 0
    b = zeros(nξ, nσ)
    χ, uξ, uη, uσ, U = inversion(b)
    iSave = 0
    saveCheckpoint2DPG(b, χ, uξ, uη, uσ, U, t, iSave)
    iSave += 1
    χEkman = getChiEkman(b)
    
    # plot initial state
    iImg = 0
    # plotCurrentState(t, χ, χEkman, uξ, uη, uσ, b, iImg)
    # iImg += 1

    # flatten for matrix mult
    bVec = reshape(b, nPts, 1)
    uξVec = reshape(uξ, nPts, 1)
    uσVec = reshape(uσ, nPts, 1)
        
    # main loop
    for i=1:nSteps
        t += Δt

        # implicit euler diffusion
        diffRHS = bVec + diffVec*Δt

        if ξVariation
            # RHS function (note the parentheses here to allow for sparse matrices to work first)

            fAdvRHS(bVec, t) = -(uξVec.*(ξDerivativeMat*bVec) + uσVec.*(σDerivativeMat*bVec) + N^2*uξVec.*HxVec.*σσVec + N^2*uσVec.*HVec)
            # fAdvRHS(bVec, t) = -(uξVec.*(ξDerivativeMat*bVec) + N^2*uξVec.*HxVec.*σσVec) # no vertical advection terms

            # explicit timestep for advection
            advRHS = RK4(t, Δt, bVec, fAdvRHS)
        else
            advRHS = -Δt*N^2*uξVec.*HxVec.*σσVec
        end

        # sum the two
        evolutionRHS = diffRHS + advRHS

        # boundary fluxes
        if bl
            evolutionRHS[bottomBdy] = -N^2 .+ χ[:, 1]./κ[:, 1].*(ξDerivativeTF(b[:, 1]) .- N^2*Hx.(ξ))
            evolutionRHS[topBdy] = χ[:, nσ]./κ[:, nσ].*ξDerivativeTF(b[:, nσ])
        else
            evolutionRHS[bottomBdy] .= -N^2
            evolutionRHS[topBdy] .= 0
        end

        # solve
        bVec = evolutionLHS\evolutionRHS

        # reshape
        b = reshape(bVec, nξ, nσ)

        # invert buoyancy for flow
        χ, uξ, uη, uσ, U = inversion(b)
        uξVec = reshape(uξ, nPts, 1)
        uσVec = reshape(uσ, nPts, 1)

        # log
        println(@sprintf("t = %.2f years (i = %d) (U = %.2e m2 s-1)", t/secsInYear, i, U))

        # # CFL stuff
        # uξCFL = minimum(abs.(dξ./uξ)) 
        # uσCFL = minimum(abs.(dσ./uσ)) 
        # println(@sprintf("CFL: uξ=%.2f days, uσ=%.2f days", uξCFL/secsInDay, uσCFL/secsInDay)) 
        
        if i % nStepsPlot == 0
            # plot flow
            χEkman = getChiEkman(b)
            plotCurrentState(t, χ, χEkman, uξ, uη, uσ, b, iImg)
            iImg += 1
        end
        if i % nStepsSave == 0
            saveCheckpoint2DPG(b, χ, uξ, uη, uσ, U, t, iSave)
            iSave += 1
        end
    end

    b = reshape(bVec, nξ, nσ)

    return b
end
