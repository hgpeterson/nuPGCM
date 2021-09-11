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
    LHS = getEvolutionLHS(m, s)

Generate the left-hand side matrix for the evolution problem with flux boundary conditions on the boundaries.
"""
function getEvolutionLHS(m::ModelSetup2DPG, s::ModelState2DPG, a::Real, b::Real)
    # bottom and top boundaries in 1D
    umap = reshape(1:m.nξ*m.nσ, m.nξ, m.nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, m.nσ]

    # LHS matrix
    LHS = a*I - b*m.Δt*m.D 

    # no flux boundaries
    LHS[bottomBdy, :] = m.D[bottomBdy, :]
    LHS[topBdy, :] = m.D[topBdy, :]

    return lu(LHS)
end

"""
    resetBCs!(m, s, RHS; bl)

Modify the right-hand side vector `RHS` to include boundary conditions at the top and bottom.
"""
function resetBCs!(m::ModelSetup2DPG, s::ModelState2DPG, RHS::Array{Float64,2}; bl=false)
    # boundary fluxes: dσ(b)/H at σ = -1, 0
    if bl
        RHS[:, 1] = s.χ[:, 1].*ξDerivative(m, s.b, 1)./m.κ[:, 1]
        RHS[:, m.nσ] = s.χ[:, m.nσ].*ξDerivative(m, s.b, m.nσ)./m.κ[:, m.nσ] .+ m.N[:, m.nσ].^2
    else
        RHS[:, 1] .= 0
        RHS[:, m.nσ] .= m.N[:, m.nσ].^2
    end
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

    # store previous buoyancy field for timestepping scheme
    nPrev = 1
    # nPrev = 3
    bPrev = zeros(nPrev, m.nξ, m.nσ)

    # get LHS matrices
    # LHS1 = getEvolutionLHS(m, s, 1, 1)
    # LHS2 = getEvolutionLHS(m, s, 3, 2)
    # LHS3 = getEvolutionLHS(m, s, 11/6, 1)
    # LHS  = getEvolutionLHS(m, s, 25/12, 1)
    LHS  = getEvolutionLHS(m, s, 1, 1/2)

    # if you want to check CFL
    dξ = m.L/m.nξ
    dσ = zeros(m.nξ, m.nσ)
    σσ = repeat(m.σ', m.nξ, 1)
    dσ[:, 1:end-1] = σσ[:, 2:end] - σσ[:, 1:end-1]
    dσ[:, end] = dσ[:, end-1]

    # main loop
    t = 0
    for i=1:nSteps
        # explicit timestep for advection
        function adv_func(b)
            χ, uξ, uη, uσ, U = invert(m, b; bl=bl)
            if m.ξVariation
                return -uξ.*ξDerivative(m, b) .- uσ.*σDerivative(m, b)
            else
                return -uσ.*σDerivative(m, b)
            end
        end

        if i == 1
            # first step: CNAB1
            RHS = s.b + m.Δt*(adv_func(s.b) + 1/2*reshape(m.D*s.b[:], m.nξ, m.nσ)) # right-hand-side
            resetBCs!(m, s, RHS; bl=bl) # modify RHS to implement boundary conditions
            bPrev[1, :, :] = s.b # store previoius step for next time
            s.b[:, :] = reshape(LHS\RHS[:], m.nξ, m.nσ)  # solve
            s.i[1] = i + 1 # next step
        else
            # other steps: CNAB2
            RHS = s.b + m.Δt*(3/2*adv_func(s.b) - 1/2*adv_func(bPrev[1, :, :]) + 1/2*reshape(m.D*s.b[:], m.nξ, m.nσ))
            resetBCs!(m, s, RHS; bl=bl)
            bPrev[1, :, :] = s.b 
            s.b[:, :] = reshape(LHS\RHS[:], m.nξ, m.nσ)
            s.i[1] = i + 1
        end
        # if i == 1
        #     # first step: SBDF1
        #     RHS = s.b + m.Δt*adv_func(s.b) # right-hand-side
        #     resetBCs!(m, s, RHS; bl=bl) # modify RHS to implement boundary conditions
        #     bPrev[1, :, :] = s.b # store previoius step for next time
        #     s.b[:, :] = reshape(LHS1\RHS[:], m.nξ, m.nσ)  # solve
        #     s.i[1] = i + 1 # next step
        # elseif i == 2
        #     # second step: SBDF2
        #     RHS = 4*s.b - bPrev[1, :, :] + 2*m.Δt*(2*adv_func(s.b) - adv_func(bPrev[1, :, :]))
        #     resetBCs!(m, s, RHS; bl=bl)
        #     bPrev[2, :, :] = bPrev[1, :, :] # move previous step back one
        #     bPrev[1, :, :] = s.b # store current step as previoius step
        #     s.b[:, :] = reshape(LHS2\RHS[:], m.nξ, m.nσ)
        #     s.i[1] = i + 1
        # elseif i == 3
        #     # second step: SBDF3
        #     RHS = 3*s.b - 3/2*bPrev[1, :, :]  + 1/3*bPrev[2, :, :] + m.Δt*(3*adv_func(s.b) - 3*adv_func(bPrev[1, :, :]) + adv_func(bPrev[2, :, :]))
        #     resetBCs!(m, s, RHS; bl=bl)
        #     bPrev[3, :, :] = bPrev[2, :, :] 
        #     bPrev[2, :, :] = bPrev[1, :, :] 
        #     bPrev[1, :, :] = s.b 
        #     s.b[:, :] = reshape(LHS3\RHS[:], m.nξ, m.nσ)
        #     s.i[1] = i + 1
        # else
        #     # other steps: SBDF4
        #     RHS = 4*s.b - 3*bPrev[1, :, :]  + 4/3*bPrev[2, :, :] - 1/4*bPrev[3, :, :] + m.Δt*(4*adv_func(s.b) - 6*adv_func(bPrev[1, :, :]) + 4*adv_func(bPrev[2, :, :]) - adv_func(bPrev[3, :, :]))
        #     resetBCs!(m, s, RHS; bl=bl)
        #     bPrev[3, :, :] = bPrev[2, :, :] 
        #     bPrev[2, :, :] = bPrev[1, :, :] 
        #     bPrev[1, :, :] = s.b 
        #     s.b[:, :] = reshape(LHS\RHS[:], m.nξ, m.nσ)
        #     s.i[1] = i + 1
        # end
        t += m.Δt

        # println(trapz(s.b[1, :], m.z[1, :]))
        # Bi = -3.9199999965243717
        # Bf = -3.9071636677290256
        # println(abs((Bi - Bf)/Bi)) # about 0.3%
        # error()

        # invert buoyancy for flow and save to state
        invert!(m, s; bl=bl)

        # log
        println(@sprintf("t = %.2f yr | i = %d | χₘₐₓ = %.2e m2 s-1", t/secsInYear, i, maximum(abs.(s.χ))))

        # # CFL stuff
        # uξCFL = minimum(abs.(dξ./s.uξ)) 
        # uσCFL = minimum(abs.(dσ./s.uσ)) 
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