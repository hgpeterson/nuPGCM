"""
    A, rhs = getSteadySystem(m)   

Compute matrix `A` and right-hand side vector `rhs` for steady 1D equations.
"""
function getSteadySystem(m::ModelSetup1D)
    nVars = 3
    nPts = nVars*m.nz

    umap = reshape(1:nPts, nVars, m.nz)    
    A = Tuple{Int64,Int64,Float64}[] # LHS matrix 
    rhs = zeros(nPts)                # RHS vector

    # Main loop, insert stencil in matrices for each node point
    for j=2:m.nz-1
        # dz stencil
        fd_z = mkfdstencil(m.z[j-1:j+1], m.z[j], 1)
        ν_z = sum(fd_z.*m.ν[j-1:j+1])
        κ_z = sum(fd_z.*m.κ[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(m.z[j-1:j+1], m.z[j], 2)

        # eqtn 1: -f*v - b*tan(θ) - dz(ν*dz(u)) = 0 
        row = umap[1, j]
        push!(A, (row, umap[2, j],   -m.f))
        push!(A, (row, umap[3, j],   -tan(m.θ)))
        push!(A, (row, umap[1, j-1], -(ν_z*fd_z[1] + m.ν[j]*fd_zz[1])))
        push!(A, (row, umap[1, j],   -(ν_z*fd_z[2] + m.ν[j]*fd_zz[2])))
        push!(A, (row, umap[1, j+1], -(ν_z*fd_z[3] + m.ν[j]*fd_zz[3])))

        # eqtn 2: f*u - dz(ν*dz(v)) = 0 
        row = umap[2, j]
        push!(A, (row, umap[1, j],   m.f))
        push!(A, (row, umap[2, j-1], -(ν_z*fd_z[1] + m.ν[j]*fd_zz[1])))
        push!(A, (row, umap[2, j],   -(ν_z*fd_z[2] + m.ν[j]*fd_zz[2])))
        push!(A, (row, umap[2, j+1], -(ν_z*fd_z[3] + m.ν[j]*fd_zz[3])))

        # eqtn 3: N^2*u*tan(θ) - dz(κ*dz(b)) = dz(κ)*N^2 
        row = umap[3, j]
        push!(A, (row, umap[1, j],   m.N2*tan(m.θ)))
        push!(A, (row, umap[3, j-1], -(κ_z*fd_z[1] + m.κ[j]*fd_zz[1])))
        push!(A, (row, umap[3, j],   -(κ_z*fd_z[2] + m.κ[j]*fd_zz[2])))
        push!(A, (row, umap[3, j+1], -(κ_z*fd_z[3] + m.κ[j]*fd_zz[3])))
        rhs[row] = κ_z*m.N2
    end

    # Boundary Conditions: Bottom
    # u = 0
    row = umap[1, 1] 
    push!(A, (row, row, 1.0))
    rhs[row] = 0
    # v = 0
    row = umap[2, 1] 
    push!(A, (row, row, 1.0))
    rhs[row] = 0 
    # dz(b) = -N^2
    row = umap[3, 1] 
    fd_z = mkfdstencil(m.z[1:3], m.z[1], 1)
    push!(A, (row, umap[3, 1], fd_z[1]))
    push!(A, (row, umap[3, 2], fd_z[2]))
    push!(A, (row, umap[3, 3], fd_z[3]))
    rhs[row]  = -m.N2 

    # Boundary Conditions: Top
    fd_z = mkfdstencil(m.z[m.nz-2:m.nz], m.z[m.nz], 1)
    # dz(u) = 0
    row = umap[1, m.nz] 
    push!(A, (row, umap[1, m.nz-2], fd_z[1]))
    push!(A, (row, umap[1, m.nz-1], fd_z[2]))
    push!(A, (row, umap[1, m.nz],   fd_z[3]))
    rhs[row] = 0
    # dz(v) = 0
    row = umap[2, m.nz] 
    push!(A, (row, umap[2, m.nz-2], fd_z[1]))
    push!(A, (row, umap[2, m.nz-1], fd_z[2]))
    push!(A, (row, umap[2, m.nz],   fd_z[3]))
    rhs[row] = 0
    # dz(b) = 0
    row = umap[3, m.nz]
    push!(A, (row, umap[3, m.nz-2], fd_z[1]))
    push!(A, (row, umap[3, m.nz-1], fd_z[2]))
    push!(A, (row, umap[3, m.nz],   fd_z[3]))
    rhs[row] = 0

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nPts, nPts)

    return A, rhs
end

"""
    b = steadyState()

Compute canonical steady state.
"""
function steadyState(m::ModelSetup1D)
    # grid points
    nVars = 3
    nPts = nVars*m.nz

    # for flattening for matrix mult
    umap = reshape(1:nPts, nVars, m.nz)

    # get matrix and right-hand side vector
    A, rhs = getSteadySystem(m)

    # solve
    sol = reshape(A\rhs, nVars, m.nz)
    u = sol[1, :]
    v = sol[2, :]
    b = sol[3, :]

    # save data
    s = ModelState1D(b, u, v, [0], [-1])
    saveState1D(s, -1)

    return s
end