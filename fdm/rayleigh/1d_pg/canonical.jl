################################################################################
# Utility functions 
################################################################################

"""
    û, v = invert(b)

Get flow from buoyancy field.
"""
function invert(b)
    û = @. b*sinθ*r/cosθ^2/(f^2 + r^2)
    v = @. -f*û*cosθ/r
    return û, v
end

"""
    u, w = rotate(û)

Rotate `û` into physical coordinate components `u` and `w`.
"""
function rotate(û)
    u = @. û*cosθ
    w = @. û*sinθ
    return u, w
end

################################################################################
# Solver functions 
################################################################################

"""
    diffMat, diffVec, bdyMat = getMatrices()   

Compute matrices for 1D equations.
"""
function getMatrices()
    nPts = nx*nz

    umap = reshape(1:nPts, nx, nz)    
    diffMat = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix 
    diffVec = zeros(nPts)                          # diffusion operator vector 
    bdyMat = Tuple{Int64,Int64,Float64}[]          # matrix for boundary conditions

    # Main loop, insert stencil in matrices for each node point
    for i=1:nx
        for j=2:nz-1
            # dẑ stencil
            fd_ẑ = mkfdstencil(ẑẑ[i, j-1:j+1], ẑẑ[i, j], 1)
            κ_ẑ = sum(fd_ẑ.*κ[i, j-1:j+1])

            # dẑẑ stencil
            fd_ẑẑ = mkfdstencil(ẑẑ[i, j-1:j+1], ẑẑ[i, j], 2)

            # b diffusion term: dẑ(κ(N^2*cos(θ) + dẑ(b))) = dẑ(κ)*N^2*cos(θ) + dẑ(κ)*dẑ(b) + κ*dẑẑ(b)
            row = umap[i, j]
            push!(diffMat, (row, umap[i, j-1], (κ_ẑ*fd_ẑ[1] + κ[i, j]*fd_ẑẑ[1])))
            push!(diffMat, (row, umap[i, j],   (κ_ẑ*fd_ẑ[2] + κ[i, j]*fd_ẑẑ[2])))
            push!(diffMat, (row, umap[i, j+1], (κ_ẑ*fd_ẑ[3] + κ[i, j]*fd_ẑẑ[3])))
            diffVec[row] = κ_ẑ*N^2*cosθ[i, j]
        end

        # Boundary Conditions: Bottom
        # dẑ(b) = -N^2*cos(θ)
        row = umap[i, 1] 
        fd_ẑ = mkfdstencil(ẑẑ[i, 1:3], ẑẑ[i, 1], 1)
        push!(bdyMat, (row, umap[i, 1], fd_ẑ[1]))
        push!(bdyMat, (row, umap[i, 2], fd_ẑ[2]))
        push!(bdyMat, (row, umap[i, 3], fd_ẑ[3]))

        # Boundary Conditions: Top
        # dẑ(b) = 0
        row = umap[i, nz]
        push!(bdyMat, (row, umap[i, nz-2], fd_ẑ[1]))
        push!(bdyMat, (row, umap[i, nz-1], fd_ẑ[2]))
        push!(bdyMat, (row, umap[i, nz],   fd_ẑ[3]))
    end

    # Create CSC sparse matrix from matrix elements
    diffMat = sparse((x->x[1]).(diffMat), (x->x[2]).(diffMat), (x->x[3]).(diffMat), nPts, nPts)
    bdyMat = sparse((x->x[1]).(bdyMat), (x->x[2]).(bdyMat), (x->x[3]).(bdyMat), nPts, nPts)

    return diffMat, diffVec, bdyMat
end

"""
    LHS = getLHS(Δt, diffMat, bdyMat, bottomBdy, topBdy)

Get implicit euler left hand side matrix.
"""
function getLHS(Δt, diffMat, bdyMat, bottomBdy, topBdy)
    # implicit euler
    LHS = I - diffMat*Δt 

    # no flux boundaries
    LHS[bottomBdy, :] = bdyMat[bottomBdy, :]
    LHS[topBdy, :] = bdyMat[topBdy, :]

    return LHS
end

"""
    sol = evolveCanonical1D(nSteps)

Solve 1D equations over 2D ridge with time.
"""
function evolveCanonical1D(nSteps)
    # grid points
    nPts = nx*nz

    # timestep
    Δt = 3*3600
    nStepsPlot = 800
    nStepsSave = 80

    # for flattening for matrix mult
    umap = reshape(1:nPts, nx, nz)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nz]

    # get matrices and vectors
    diffMat, diffVec, bdyMat = getMatrices()

    # left-hand side for evolution equation (save LU decomposition for speed)
    LHS = lu(getLHS(Δt, diffMat, bdyMat, bottomBdy, topBdy))

    # initial condition
    t = 0
    b = zeros(nx, nz)
    #= # load data =#
    #= file = h5open("b.h5", "r") =#
    #= b = read(file, "b") =#
    #= t = read(file, "t") =#
    #= close(file) =#

    # plot initial state
    û, v = invert(b)
    iImg = 0
    plotCurrentState(t, û, v, b, iImg)

    # flatten for matrix mult
    bVec = reshape(b, nPts, 1)
    sinθVec = reshape(sinθ, nPts, 1)
    cosθVec = reshape(cosθ, nPts, 1)

    # main loop
    for i=1:nSteps
        t += Δt
        tDays = t/86400

        # implicit euler diffusion
        diffRHS = bVec + diffVec*Δt

        # function to compute explicit RHS
        fExplicit(bVec, t) = @. -r*N^2*sinθVec^2*bVec/cosθVec^2/(f^2 + r^2)

        # explicit timestep for RHS
        explicitRHS = RK4(t, Δt, bVec, fExplicit)

        # sum the two
        RHS = diffRHS + explicitRHS

        # boundary conditions
        RHS[bottomBdy]  = -N^2*cosθ[:, 1] # b flux bot
        RHS[topBdy] .= 0 # b flux top

        # solve
        bVec = LHS\RHS

        # log
        println(@sprintf("t = %.2f days (i = %d)", tDays, i))
        if i % nStepsPlot == 0
            # reshape and invert
            b = reshape(bVec, nx, nz)
            û, v = invert(b)

            # plot flow
            iImg += 1
            plotCurrentState(t, û, v, b, iImg)
        end
        if i % nStepsSave == 0
            # reshape
            b = reshape(bVec, nx, nz)

            # save data
            filename = @sprintf("b%d.h5", tDays)
            println("saving ", filename)
            file = h5open(filename, "w")
            write(file, "b", b)
            write(file, "t", t)
            close(file)
        end
    end

    b = reshape(bVec, nx, nz)

    return b
end

"""
    b = steadyState()

Solve 1D equations over 2D ridge to steady state.
"""
function steadyState()
    # grid points
    nPts = nx*nz

    # for flattening for matrix mult
    umap = reshape(1:nPts, nx, nz)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, nz]
    sinθVec = reshape(sinθ, nPts, 1)
    cosθVec = reshape(cosθ, nPts, 1)

    # get matrices and vectors
    diffMat, diffVec, bdyMat = getMatrices()

    # LHS
    α = -r*N^2*sinθVec.^2 ./cosθVec.^2/(f^2 + r^2)
    α = Diagonal(α[:, 1])
    LHS = α + diffMat

    # boundaries
    LHS[bottomBdy, :] = bdyMat[bottomBdy, :]
    LHS[topBdy, :] = bdyMat[topBdy, :]

    # RHS
    RHS = -diffVec
    # boundaries
    RHS[bottomBdy]  .= -N^2*cosθ[:, 1] # b flux bot
    RHS[topBdy] .= 0    # b flux top

    # solve
    bVec = LHS\RHS

    # reshape and invert
    b = reshape(bVec, nx, nz)
    û, v = invert(b)

    # save data
    filename = "bSteady.h5"
    println("saving ", filename)
    file = h5open(filename, "w")
    write(file, "b", b)
    write(file, "t", Inf)
    close(file)

    # plot flow
    plotCurrentState(Inf, û, v, b, 999)

    return b
end
