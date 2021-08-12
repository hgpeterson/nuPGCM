################################################################################
# PG evolution functions
################################################################################

using Printf

"""
    D = getDiffusionMatrix(ξ, σ, κ, H)

Compute the matrices needed for evolution equation integration.
"""
function getDiffusionMatrix(ξ::Array{Float64,1}, σ::Array{Float64,1}, κ::Array{Float64,2},  H::Array{Float64,1})
    nξ = size(ξ, 1)
    nσ = size(σ, 1)
    nPts = nξ*nσ

    umap = reshape(1:nPts, nξ, nσ)    
    D = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix (with boundary flux conditions)

    # Main loop, insert stencil in matrices for each node point
    for i=1:nξ
        # interior nodes only for operators
        for j=2:nσ-1
            row = umap[i, j] 

            # dσ stencil
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            κ_σ = sum(fd_σ.*κ[i, j-1:j+1])

            # dσσ stencil
            fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

            # diffusion term: dσ(κ*dσ(b))/H^2 = 1/H^2*(dσ(κ)*dσ(b) + κ*dσσ(b))
            push!(D, (row, umap[i, j-1], (κ_σ*fd_σ[1] + κ[i, j]*fd_σσ[1])/H[i]^2))
            push!(D, (row, umap[i, j],   (κ_σ*fd_σ[2] + κ[i, j]*fd_σσ[2])/H[i]^2))
            push!(D, (row, umap[i, j+1], (κ_σ*fd_σ[3] + κ[i, j]*fd_σσ[3])/H[i]^2))
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, 1], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, 2], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, 3], fd_σ[3]/H[i]))

        # flux at boundaries: top
        row = umap[i, nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, nσ-2], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, nσ-1], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, nσ],   fd_σ[3]/H[i]))
    end

    # Create CSC sparse matrix from matrix elements
    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), nPts, nPts)

    return D
end

"""
    evolutionLHS = getEvolutionLHS(nξ, nσ, D, Δt)

Generate the left-hand side matrix for the evolution problem of the form `I - D*Δt`
and flux boundary conditions on the boundaries.
"""
function getEvolutionLHS(nξ::Int64, nσ::Int64, D::SparseMatrixCSC{Float64,Int64}, Δt::Real)
    # implicit euler
    evolutionLHS = I - D*Δt 

    # bottom and top boundaries in 1D
    umap = reshape(1:nξ*nσ, nξ, nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nσ]

    # no flux boundaries
    evolutionLHS[bottomBdy, :] = D[bottomBdy, :]
    evolutionLHS[topBdy, :] = D[topBdy, :]

    return lu(evolutionLHS)
end

"""
    evolve!(m, s, tFinal, tPlot, tSave; bl=false)

Solve evoluion equation for `b` and update model state.
If `bl` set to `true`, use boundary layer theory inversion and boundary conditions. 
"""
function evolve!(m::ModelSetup2DPG, s::ModelState2DPG, tFinal::Real, tPlot::Real, tSave::Real; bl=false)
    # grid points
    nPts = m.nξ*m.nσ

    # timestep
    nSteps = Int64(tFinal/m.Δt)
    nStepsPlot = Int64(tPlot/m.Δt)
    nStepsSave = Int64(tSave/m.Δt)

    # save initial state
    iSave = 0
    saveState2DPG(s, iSave)
    iSave += 1
    
    # plot initial state
    iImg = 0
    plotCurrentState(m, s, iImg)
    iImg += 1

    # main loop
    t = 0
    for i=1:nSteps
        t += m.Δt

        # implicit euler diffusion
        diffRHS = s.b

        # explicit timestep for advection
        # χ, uξ, uη, uσ, U = invert(m, s.b; bl=bl) # for speed
        function fAdvRHS(b, t)
            χ, uξ, uη, uσ, U = invert(m, b; bl=bl) # for accuracy
            if m.ξVariation
                return -uξ.*ξDerivative(m, b) .- uσ.*σDerivative(m, b)
            else
                return -uσ.*σDerivative(m, b)
            end
        end
        advRHS = RK4(t, m.Δt, s.b, fAdvRHS)

        # sum the two
        evolutionRHS = diffRHS .+ advRHS

        # boundary fluxes: dσ(b)/H at σ = -1, 0
        if bl
            evolutionRHS[:, 1] = s.χ[:, 1].*ξDerivative(m, s.b[:, 1])./m.κ[:, 1]
            evolutionRHS[:, m.nσ] = s.χ[:, m.nσ].*ξDerivative(m, s.b[:, m.nσ])./m.κ[:, m.nσ] .+ m.N^2
        else
            evolutionRHS[:, 1] .= 0
            evolutionRHS[:, m.nσ] .= m.N^2
        end

        # solve and update model state
        s.b[:, :] = reshape(m.evolutionLHS\evolutionRHS[:], m.nξ, m.nσ)
        s.i[1] = i + 1

        # invert buoyancy for flow and save to state
        invert!(m, s; bl=bl)

        # log
        println(@sprintf("t = %.2f years (i = %d)", t/secsInYear, i))

        # # CFL stuff
        # uξCFL = minimum(abs.(dξ./uξ)) 
        # uσCFL = minimum(abs.(dσ./uσ)) 
        # println(@sprintf("CFL: uξ=%.2f days, uσ=%.2f days", uξCFL/secsInDay, uσCFL/secsInDay)) 
        
        if i % nStepsPlot == 0
            # plot flow
            plotCurrentState(m, s, iImg)
            iImg += 1
        end
        if i % nStepsSave == 0
            saveState2DPG(s, iSave)
            iSave += 1
        end
    end
end