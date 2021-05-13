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
        # flux term: dσ(b)/H = -N^2
        push!(bdyFluxMat, (row, umap[i, 1], fd_σ[1]/H(ξ[i])))
        push!(bdyFluxMat, (row, umap[i, 2], fd_σ[2]/H(ξ[i])))
        push!(bdyFluxMat, (row, umap[i, 3], fd_σ[3]/H(ξ[i])))

        # flux at boundaries: top
        row = umap[i, nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        # flux term: dσ(b)/H = -N^2
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
    b = evolve(tFinal)

Solve full nonlinear equation for `b` for `tFinal` seconds.
"""
function evolve(tFinal)
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
    χ, uξ, uη, uσ, U = invert(b)
    iSave = 0
    saveCheckpointTF(b, χ, uξ, uη, uσ, U, t, iSave)
    iSave += 1
    χEkman = getChiEkman(b)
    
    # plot initial state of all zeros and no flow
    iImg = 0
    plotCurrentState(t, χ, χEkman, uξ, uη, uσ, b, iImg)
    iImg += 1

    # flatten for matrix mult
    bVec = reshape(b, nPts, 1)
    uξVec = reshape(uξ, nPts, 1)
    uσVec = reshape(uσ, nPts, 1)
        
    # main loop
    for i=1:nSteps
        t += Δt
        tDays = t/86400

        # implicit euler diffusion
        diffRHS = bVec + diffVec*Δt

        if ξVariation
            # RHS function (note the parentheses here to allow for sparse matrices to work first)
            fAdvRHS(bVec, t) = -(uξVec.*(ξDerivativeMat*bVec) + uσVec.*(σDerivativeMat*bVec) + N^2*uξVec.*HxVec.*σσVec + N^2*uσVec.*HVec)

            # explicit timestep for advection
            advRHS = RK4(t, Δt, bVec, fAdvRHS)
        else
            advRHS = -Δt*N^2*uξVec.*HxVec.*σσVec
        end

        # sum the two
        evolutionRHS = diffRHS + advRHS

        # boundary fluxes
        evolutionRHS[bottomBdy] .= -N^2
        evolutionRHS[topBdy] .= 0
        #= evolutionRHS[topBdy] .= -N^2 =#

        # solve
        bVec = evolutionLHS\evolutionRHS

        # log
        println(@sprintf("t = %.2f days (i = %d)", tDays, i))

        # reshape
        b = reshape(bVec, nξ, nσ)

        # invert buoyancy for flow
        χ, uξ, uη, uσ, U = invert(b)
        uξVec = reshape(uξ, nPts, 1)
        uσVec = reshape(uσ, nPts, 1)

        #= # CFL stuff =#
        #= uξCFL = minimum(abs.(dξ./uξ)) =#
        #= uσCFL = minimum(abs.(dσ./uσ)) =#
        #= println(@sprintf("CFL uξ: %.2f days", uξCFL/86400)) =#
        #= println(@sprintf("CFL uσ: %.2f days", uσCFL/86400)) =#
        
        if i % nStepsPlot == 0
            # plot flow
            χEkman = getChiEkman(b)
            plotCurrentState(t, χ, χEkman, uξ, uη, uσ, b, iImg)
            iImg += 1
        end
        if i % nStepsSave == 0
            saveCheckpointTF(b, χ, uξ, uη, uσ, U, t, iSave)
            iSave += 1
        end
    end

    b = reshape(bVec, nξ, nσ)

    return b
end

#= function advTest(nSteps) =#
#=     # grid points =#
#=     nPts = nξ*nσ =#

#=     # timestep =#
#=     Δt = 10*86400 =#
#=     nStepsPlot = 10 =#

#=     # adv speed =#
#=     #1= uξ = dξ/Δt =1# =#
#=     uξ = 1e-2 =#
#=     #1= uσ = -minimum(dσ)/Δt =1# =#
#=     uσ = -3e-9 =#

#=     println("uξCFL: ", dξ/abs(uξ)) =#
#=     println("uσCFL: ", minimum(dσ)/abs(uσ)) =#

#=     # initial condition =#
#=     t = 0 =#
#=     b0(ξ, σ) = exp(-(ξ - L/2)^2/(L/8)^2 - (σ + 1/2)^2/(1/4)^2) =#
#=     bExact(ξ, σ, t) = b0(ξ - uξ*t, σ - uσ*t) =#
#=     b = @. b0(ξξ, σσ) =#

#=     # plot initial state of all zeros and no flow =#
#=     iImg = 0 =#
#=     ridgePlot2Fields(b, bExact.(ξξ, σσ, 0), "", "") =#
#=     savefig(@sprintf("b%03d.png", iImg)) =#
#=     close() =#

#=     # main loop =#
#=     for i=1:nSteps =#
#=         t += Δt =#
#=         tDays = t/86400 =#

#=         # function to compute advection RHS =#
#=         # (note the parentheses here to allow for sparse matrices to work first) =#
#=         fAdvRHS(b) = -uξ*ξDerivativeTF(b) - uσ*σDerivativeTF(b) =#

#=         # explicit timestep for advection =#
#=         advRHS = explicitRHS(Δt, b, fAdvRHS) =#

#=         # solve =#
#=         b += advRHS =#

#=         # log =#
#=         println(@sprintf("t = %.2f days (i = %d)", tDays, i)) =#
#=         if i % nStepsPlot == 0 =#
#=             # plot flow =#
#=             iImg += 1 =#
#=             ridgePlot2Fields(b, bExact.(ξξ, σσ, t), "", "") =#
#=             savefig(@sprintf("b%03d.png", iImg)) =#
#=             close() =#
#=         end =#
#=     end =#
#=     return b =#
#= end =#
