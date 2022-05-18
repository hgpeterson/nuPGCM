"""
    A, diffVec = get_A_diffVec(m)   

Compute matrix and vector such that such that sol_t = A*sol + diffVec
where sol = (u, v, b, ∂ₓP). `A` also contains the proper boundary conditions.
"""
function get_A_diffVec(m::ModelSetup1D)
    nVars = 3
    nPts = nVars*m.nz + 1

    umap = reshape(1:(nPts-1), nVars, m.nz)    
    A = Tuple{Int64,Int64,Float64}[]  
    diffVec = zeros(nPts)

    # Main loop, insert stencil in matrices for each node point
    for j=2:m.nz-1
        # dz stencil
        fd_z = mkfdstencil(m.z[j-1:j+1], m.z[j], 1)
        ν_z = sum(fd_z.*m.ν[j-1:j+1])
        κ_z = sum(fd_z.*m.κ[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(m.z[j-1:j+1], m.z[j], 2)

        # 1st eqtn: u_t = f*v - ∂ₓP + b*tan(θ) + (ν u_z)_z
        row = umap[1, j]
        # first term
        push!(A, (row, umap[2, j], m.f))
        # second term
        push!(A, (row, nPts, -1))
        # third term
        push!(A, (row, umap[3, j], tan(m.θ)))
        # fourth term: dz(ν*dz(u))) = dz(ν)*dz(u) + ν*dzz(u)
        push!(A, (row, umap[1, j-1], ν_z*fd_z[1] + m.ν[j]*fd_zz[1]))
        push!(A, (row, umap[1, j],   ν_z*fd_z[2] + m.ν[j]*fd_zz[2]))
        push!(A, (row, umap[1, j+1], ν_z*fd_z[3] + m.ν[j]*fd_zz[3]))

        # 2nd eqtn: v_t = -f*u + (ν*v_z)_z
        row = umap[2, j]
        # first term:
        push!(A, (row, umap[1, j], -m.f))
        # second term: dz(ν*dz(v))) = dz(ν)*dz(v) + ν*dzz(v)
        push!(A, (row, umap[2, j-1], ν_z*fd_z[1] + m.ν[j]*fd_zz[1]))
        push!(A, (row, umap[2, j],   ν_z*fd_z[2] + m.ν[j]*fd_zz[2]))
        push!(A, (row, umap[2, j+1], ν_z*fd_z[3] + m.ν[j]*fd_zz[3]))

        # 3rd eqtn: b_t = -N^2*u*tan(θ) + [κ*(N^2 + b_z)]_z
        row = umap[3, j]
        # first term
        push!(A, (row, umap[1, j], -m.N2*tan(m.θ)))
        # second term: dz(κ(1 + dz(b))) = dz(κ) + dz(κ)*dz(b) + κ*dzz(b)
        push!(A, (row, umap[3, j-1], (κ_z*fd_z[1] + m.κ[j]*fd_zz[1])))
        push!(A, (row, umap[3, j],   (κ_z*fd_z[2] + m.κ[j]*fd_zz[2])))
        push!(A, (row, umap[3, j+1], (κ_z*fd_z[3] + m.κ[j]*fd_zz[3])))
        diffVec[row] = m.N2*κ_z
    end

    # Boundary Conditions: Bottom
    # u = 0
    row = umap[1, 1] 
    push!(A, (row, row, 1.0))
    # v = 0
    row = umap[2, 1] 
    push!(A, (row, row, 1.0))
    # b_z = -N^2
    row = umap[3, 1] 
    fd_z = mkfdstencil(m.z[1:3], m.z[1], 1)
    push!(A, (row, umap[3, 1], fd_z[1]))
    push!(A, (row, umap[3, 2], fd_z[2]))
    push!(A, (row, umap[3, 3], fd_z[3]))

    # Boundary Conditions: Top
    fd_z = mkfdstencil(m.z[m.nz-2:m.nz], m.z[m.nz], 1)
    # dz(u) = 0
    row = umap[1, m.nz] 
    push!(A, (row, umap[1, m.nz-2], fd_z[1]))
    push!(A, (row, umap[1, m.nz-1], fd_z[2]))
    push!(A, (row, umap[1, m.nz],   fd_z[3]))
    # dz(v) = 0
    row = umap[2, m.nz] 
    push!(A, (row, umap[2, m.nz-2], fd_z[1]))
    push!(A, (row, umap[2, m.nz-1], fd_z[2]))
    push!(A, (row, umap[2, m.nz],   fd_z[3]))
    # dz(b) = 0
    row = umap[3, m.nz]
    push!(A, (row, umap[3, m.nz-2], fd_z[1]))
    push!(A, (row, umap[3, m.nz-1], fd_z[2]))
    push!(A, (row, umap[3, m.nz],   fd_z[3]))

    # transport constraint
    row = nPts
    if m.transportConstraint
        # transport-constrained 1D: ∂ₓP such that ∫ u dz = U
        for j=1:m.nz-1
            # trapezoidal rule: (u_j+1 + u_j)/2 * Δz_j
            push!(A, (row, umap[1, j],   (m.z[j+1] - m.z[j])/2))
            push!(A, (row, umap[1, j+1], (m.z[j+1] - m.z[j])/2))
        end
    else
        # canonical 1D: ∂ₓP = constant in time
        push!(A, (row, nPts, 1.0))
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nPts, nPts)

    return A, diffVec
end

"""
    LHS, RHS, diffVec = getTimestepMaticesVector(m)

Get left- and right-hand-side matrices for time stepping where
    (x^(n+1) - x^n)/Δt = 0.5*A*x^n + 0.5*A*x^(n+1) + diffVec
where A is the matrix such that x_t = A*x + diffVec. Returns
    LHS = I/Δt - 0.5*A,
    RHS = I/Δt + 0.5*A.
    diffVec.
"""
function getTimestepMaticesVector(m::ModelSetup1D)
    # left-hand side for evolution equation
    A, diffVec = get_A_diffVec(m)

    # grid points
    nVars = 3
    nPts = nVars*m.nz + 1

    # bottom and top
    umap = reshape(1:(nPts-1), nVars, m.nz)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, m.nz]

    # (x^(n+1) - x^n)/Δt = 0.5*A*x^n + 0.5*A*x^(n+1)
    LHS = I/m.Δt - 0.5*A 
    RHS = I/m.Δt + 0.5*A 

    # boundary conditions and integral constraint
    LHS[bottomBdy, :] = A[bottomBdy, :]
    LHS[topBdy, :] = A[topBdy, :]
    LHS[nPts, :] = A[nPts, :]

    # LU factor LHS
    LHS = lu(LHS)

    return LHS, RHS, diffVec
end

"""
    evolve!(m, s, tFinal, tSave; bl=bl)

Solve for `tFinal` seconds.
"""
function evolve!(m::ModelSetup1D, s::ModelState1D, tFinal::Real, tSave::Real)
    # grid points
    nVars = 3
    nPts = nVars*m.nz + 1
    umap = reshape(1:(nPts-1), nVars, m.nz)    

    # timestep
    nSteps = Int64(tFinal/m.Δt)
    nStepsSave = Int64(tSave/m.Δt)

    # get matrices and vector for timestepping
    LHS, RHS, diffVec = getTimestepMaticesVector(m)

    # initial condition
    iSave = 0
    saveState1D(s, iSave)
    iSave += 1

    # sol vector
    sol = zeros(nPts)
    sol[umap[1, :]] = s.u[:]
    sol[umap[2, :]] = s.v[:]
    sol[umap[3, :]] = s.b[:]
    sol[nPts] = s.∂ₓP[1]

    # main loop
    t = m.Δt
    for i=1:nSteps
        # impose tidally varying U
        if m.Uamp > 0
            m.U[1] = m.Uamp*sin(2*π*t/m.Uper)
        end

        # right-hand-side as a vector
        RHSVec = RHS*sol + diffVec

        # boundary conditions
        RHSVec[umap[1, 1]]  = 0     # u = 0 bot
        RHSVec[umap[1, m.nz]] = 0   # u decay top
        RHSVec[umap[2, 1]]  = 0     # v = 0 bot
        RHSVec[umap[2, m.nz]] = 0   # v decay top
        RHSVec[umap[3, 1]]  = -m.N2 # b flux bot
        RHSVec[umap[3, m.nz]] = 0   # b flux top
        if m.transportConstraint 
            RHSVec[nPts] = m.U[1] # ∫ u dz = U 
        else
            RHSVec[nPts] = 0 # ∂ₓP = 0
        end

        # solve
        sol = LHS\RHSVec

        # unpack
        s.u[:] = sol[umap[1, :]]
        s.v[:] = sol[umap[2, :]]
        s.b[:] = sol[umap[3, :]]
        s.∂ₓP[1] = sol[nPts]

        if i % nStepsSave == 0
            # log
            println(@sprintf("t = %.2f years (i = %d)", m.Δt*i/secsInYear, i))
            
            # save
            saveState1D(s, iSave)

            # # plot
            # setupFile = string(outFolder, "setup.h5")
            # stateFile = @sprintf("%sstate%d.h5", outFolder, iSave)
            # imgFile = @sprintf("%sprofiles_%03d.png", outFolder, iSave)
            # profilePlot(setupFile, stateFile, imgFile)

            # next
            iSave += 1
        end

        # step
        s.i[1] = i + 1
        t += m.Δt
    end
end
