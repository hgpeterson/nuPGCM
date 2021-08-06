################################################################################
# PG evolution functions
################################################################################

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

    # save initial state
    iSave = 0
    saveCheckpoint2DPG(s, iSave)
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
            saveCheckpoint2DPG(s, iSave)
            iSave += 1
        end
    end
end