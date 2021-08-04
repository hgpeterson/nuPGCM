"""
    matrices = getEvolutionMatrices(m)

Compute the matrices needed for evolution equation integration.
"""
function getEvolutionMatrices(m::ModelSetup)
    nPts = m.nξ*m.nσ

    umap = reshape(1:nPts, m.nξ, m.nσ)    
    diffMat = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix 
    bdyFluxMat = Tuple{Int64,Int64,Float64}[]      # flux at boundary matrix
    ξDerivativeMat = Tuple{Int64,Int64,Float64}[]  # advection operator matrix (ξ)
    σDerivativeMat = Tuple{Int64,Int64,Float64}[]  # advection operator matrix (σ)

    # Main loop, insert stencil in matrices for each node point
    for i=1:m.nξ
        # periodic in ξ
        iL = mod1(i-1, m.nξ)
        iR = mod1(i+1, m.nξ)

        # interior nodes only for operators
        for j=2:m.nσ-1
            row = umap[i, j] 

            # dσ stencil
            fd_σ = mkfdstencil(m.σ[j-1:j+1], m.σ[j], 1)
            κ_σ = sum(fd_σ.*m.κ[i, j-1:j+1])

            # dσσ stencil
            fd_σσ = mkfdstencil(m.σ[j-1:j+1], m.σ[j], 2)

            # diffusion term: dσ(κ*dσ(b))/H^2 = 1/H^2*(dσ(κ)*dσ(b) + κ*dσσ(b))
            push!(diffMat, (row, umap[i, j-1], (κ_σ*fd_σ[1] + m.κ[i, j]*fd_σσ[1])/m.H[i]^2))
            push!(diffMat, (row, umap[i, j],   (κ_σ*fd_σ[2] + m.κ[i, j]*fd_σσ[2])/m.H[i]^2))
            push!(diffMat, (row, umap[i, j+1], (κ_σ*fd_σ[3] + m.κ[i, j]*fd_σσ[3])/m.H[i]^2))

            # ξ advection term: dξ()
            push!(ξDerivativeMat, (row, umap[iR, j],  1.0/(2*(m.ξ[2] - m.ξ[1]))))
            push!(ξDerivativeMat, (row, umap[iL, j], -1.0/(2*(m.ξ[2] - m.ξ[1]))))

            # σ advection term: dσ()
            push!(σDerivativeMat, (row, umap[i, j-1], fd_σ[1]))
            push!(σDerivativeMat, (row, umap[i, j],   fd_σ[2]))
            push!(σDerivativeMat, (row, umap[i, j+1], fd_σ[3]))
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dσ stencil
        fd_σ = mkfdstencil(m.σ[1:3], m.σ[1], 1)
        # flux term: dσ(b)/H = ...
        push!(bdyFluxMat, (row, umap[i, 1], fd_σ[1]/m.H[i]))
        push!(bdyFluxMat, (row, umap[i, 2], fd_σ[2]/m.H[i]))
        push!(bdyFluxMat, (row, umap[i, 3], fd_σ[3]/m.H[i]))

        # flux at boundaries: top
        row = umap[i, m.nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(m.σ[m.nσ-2:m.nσ], m.σ[m.nσ], 1)
        # flux term: dσ(b)/H = ...
        push!(bdyFluxMat, (row, umap[i, m.nσ-2], fd_σ[1]/m.H[i]))
        push!(bdyFluxMat, (row, umap[i, m.nσ-1], fd_σ[2]/m.H[i]))
        push!(bdyFluxMat, (row, umap[i, m.nσ],   fd_σ[3]/m.H[i]))
    end

    # Create CSC sparse matrix from matrix elements
    diffMat = sparse((x->x[1]).(diffMat), (x->x[2]).(diffMat), (x->x[3]).(diffMat), nPts, nPts)
    bdyFluxMat = sparse((x->x[1]).(bdyFluxMat), (x->x[2]).(bdyFluxMat), (x->x[3]).(bdyFluxMat), nPts, nPts)
    ξDerivativeMat = sparse((x->x[1]).(ξDerivativeMat), (x->x[2]).(ξDerivativeMat), (x->x[3]).(ξDerivativeMat), nPts, nPts)
    σDerivativeMat = sparse((x->x[1]).(σDerivativeMat), (x->x[2]).(σDerivativeMat), (x->x[3]).(σDerivativeMat), nPts, nPts)

    return diffMat, bdyFluxMat, ξDerivativeMat, σDerivativeMat
end

"""
    evolutionLHS = getEvolutionLHS(Δt, diffMat, bdyFluxMat, bottomBdy, topBdy)

Generate the left-hand side matrix for the evolution problem of the form `I - diffmat*Δt`
and the not flux boundary condition applied to the boundaries
"""
function getEvolutionLHS(Δt::Real, diffMat::SparseMatrixCSC{Float64,Int64}, bdyFluxMat::SparseMatrixCSC{Float64,Int64}, bottomBdy::Array{Int64,1}, topBdy::Array{Int64,1})
    # implicit euler
    evolutionLHS = I - diffMat*Δt 

    # no flux boundaries
    evolutionLHS[bottomBdy, :] = bdyFluxMat[bottomBdy, :]
    evolutionLHS[topBdy, :] = bdyFluxMat[topBdy, :]

    return evolutionLHS
end

"""
    evolve!(m, s, tFinal, tPlot, tSave; bl=false)

Solve evoluion equation for `b` and update model state.
If `bl` set to `true`, use boundary layer theory inversion and boundary conditions. 
"""
function evolve!(m::ModelSetup, s::ModelState, tFinal::Real, tPlot::Real, tSave::Real; bl=false)
    # grid points
    nPts = m.nξ*m.nσ

    # timestep
    nSteps = Int64(tFinal/m.Δt)
    nStepsPlot = Int64(tPlot/m.Δt)
    nStepsSave = Int64(tSave/m.Δt)

    # for flattening for matrix mult
    umap = reshape(1:nPts, m.nξ, m.nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, m.nσ]

    # get matrices and vectors
    diffMat, bdyFluxMat, ξDerivativeMat, σDerivativeMat = getEvolutionMatrices(m)

    # left-hand side for evolution equation (save LU decomposition for speed)
    evolutionLHS = lu(getEvolutionLHS(m.Δt, diffMat, bdyFluxMat, bottomBdy, topBdy))

    # # save initial state
    # iSave = 0
    # saveCheckpoint2DPG(s, iSave)
    # iSave += 1
    
    # # plot initial state
    # iImg = 0
    # plotCurrentState(m, s, iImg)
    # iImg += 1

    # main loop
    t = 0
    for i=1:nSteps
        t += m.Δt

        # implicit euler diffusion
        diffRHS = s.b

        # explicit timestep for advection
        χ, uξ, uη, uσ, U = invert(m, s.b; bl=bl)
        function fAdvRHS(b, t)
            # χ, uξ, uη, uσ, U = invert(m, b; bl=bl)
            if m.ξVariation
                return reshape(-uξ[:].*(ξDerivativeMat*b[:]) .- uσ[:].*(σDerivativeMat*b[:]), m.nξ, m.nσ)
            else
                return reshape(-uσ[:].*(σDerivativeMat*b[:]), m.nξ, m.nσ)
            end
        end
        advRHS = RK4(t, m.Δt, s.b, fAdvRHS)

        # sum the two
        evolutionRHS = diffRHS .+ advRHS

        # boundary fluxes
        if bl
            evolutionRHS[bottomBdy] = m.H./m.κ[:, 1].*s.χ[:, 1].*ξDerivativeTF(m, s.b[:, 1])
            evolutionRHS[topBdy] = m.H./m.κ[:, m.nσ].*(m.N^2 .+ s.χ[:, m.nσ].*ξDerivativeTF(m, s.b[:, m.nσ]))
        else
            evolutionRHS[bottomBdy] .= 0
            evolutionRHS[topBdy] .= m.N^2
        end

        # solve and update model state
        s.b[:, :] = reshape(evolutionLHS\evolutionRHS[:], m.nξ, m.nσ)
        s.i[1] = i + 1

        # invert buoyancy for flow and save to state
        invert!(m, s; bl=bl)

        # log
        println(@sprintf("t = %.2f years (i = %d)", t/secsInYear, i))

        # # CFL stuff
        # uξCFL = minimum(abs.(dξ./uξ)) 
        # uσCFL = minimum(abs.(dσ./uσ)) 
        # println(@sprintf("CFL: uξ=%.2f days, uσ=%.2f days", uξCFL/secsInDay, uσCFL/secsInDay)) 
        
        # if i % nStepsPlot == 0
        #     # plot flow
        #     plotCurrentState(m, s, iImg)
        #     iImg += 1
        # end
        # if i % nStepsSave == 0
        #     saveCheckpoint2DPG(s, iSave)
        #     iSave += 1
        # end
    end
end