"""
    LHS = getInversionLHS(ν, z, f, θ)

Setup left hand side of linear system for problem.
"""
function getInversionLHS(ν::Array{Float64,1}, z::Array{Float64,1}, f::Float64, θ::Float64, transportConstraint::Bool)
    nz = size(z, 1)
    iU = nz + 1
    A = Tuple{Int64,Int64,Float64}[]  

    # for finite difference on the top and bottom boundary
    fd_bot_z =  mkfdstencil(z[1:3], z[1],  1)
    fd_top_zz = mkfdstencil(z[nz-3:nz], z[nz], 2)

    # Lower boundary conditions 
    # b.c. 1: dz(χ) = 0
    push!(A, (1, 1, fd_bot_z[1]))
    push!(A, (1, 2, fd_bot_z[2]))
    push!(A, (1, 3, fd_bot_z[3]))
    # b.c. 2: χ = 0 
    push!(A, (2, 1, 1.0))

    # Upper boundary conditions
    # b.c. 1: dzz(χ) = 0 
    push!(A, (nz, nz-3, fd_top_zz[1]))
    push!(A, (nz, nz-2, fd_top_zz[2]))
    push!(A, (nz, nz-1, fd_top_zz[3]))
    push!(A, (nz, nz,   fd_top_zz[4]))
    # b.c. 2: χ - U = 0
    push!(A, (nz-1, nz,  1.0))
    push!(A, (nz-1, iU, -1.0))

    # Interior nodes
    for j=3:nz-2
        row = j 

        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)
        ν_zz = sum(fd_zz.*ν[j-1:j+1])

        # dzzz stencil
        fd_zzz = mkfdstencil(z[j-2:j+2], z[j], 3)

        # dzzzz stencil
        fd_zzzz = mkfdstencil(z[j-2:j+2], z[j], 4)
        
        # eqtn: dzz(nu*dzz(χ)) + f^2*(χ - U)/nu = -dz(b)*tan(θ)
        # term 1 (product rule)
        push!(A, (row, j-1, ν_zz*fd_zz[1]))
        push!(A, (row, j,   ν_zz*fd_zz[2]))
        push!(A, (row, j+1, ν_zz*fd_zz[3]))

        push!(A, (row, j-2, 2*ν_z*fd_zzz[1]))
        push!(A, (row, j-1, 2*ν_z*fd_zzz[2]))
        push!(A, (row, j,   2*ν_z*fd_zzz[3]))
        push!(A, (row, j+1, 2*ν_z*fd_zzz[4]))
        push!(A, (row, j+2, 2*ν_z*fd_zzz[5]))

        push!(A, (row, j-2, ν[j]*fd_zzzz[1]))
        push!(A, (row, j-1, ν[j]*fd_zzzz[2]))
        push!(A, (row, j,   ν[j]*fd_zzzz[3]))
        push!(A, (row, j+1, ν[j]*fd_zzzz[4]))
        push!(A, (row, j+2, ν[j]*fd_zzzz[5]))
        # term 2
        push!(A, (row, j,   f^2/(ν[j])))
        push!(A, (row, iU, -f^2/(ν[j])))
    end

    # if dx(p) ~ 0 then 
    #   (1) U = U₀
    #       for transport-constrained 1D solution
    #   (2) dz(nu*dzz(χ)) = Hx*b at bottom
    #       for canonical 1D solution
    row = iU
    if transportConstraint
        push!(A, (row, row, 1.0))
    else
        # dz stencil
        fd_z = mkfdstencil(z[1:3], z[1], 1)
        ν_z = sum(fd_z.*ν[1:3])

        # dzz stencil
        fd_zz = mkfdstencil(z[1:4], z[1], 2)

        # dzzz stencil
        fd_zzz = mkfdstencil(z[1:5], z[1], 3)

        # product rule
        push!(A, (row, 1, ν_z*fd_zz[1]))
        push!(A, (row, 2, ν_z*fd_zz[2]))
        push!(A, (row, 3, ν_z*fd_zz[3]))
        push!(A, (row, 4, ν_z*fd_zz[4]))

        push!(A, (row, 1, ν[1]*fd_zzz[1]))
        push!(A, (row, 2, ν[1]*fd_zzz[2]))
        push!(A, (row, 3, ν[1]*fd_zzz[3]))
        push!(A, (row, 4, ν[1]*fd_zzz[4]))
        push!(A, (row, 5, ν[1]*fd_zzz[5]))
    end
    
    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nz+1, nz+1)

    return lu(A)
end

"""
    RHS = getInversionRHS(m, b)

Setup right hand side of linear system for problem.
"""
function getInversionRHS(m::ModelSetup1DPG, b::Array{Float64,1})
    # last row is for U
    rhs = zeros(m.nz+1)
    iU = m.nz + 1

    # eqtn: dzz(nu*dzz(χ)) + f^2(χ - U)/nu = -dz(b)*tan(θ)
    rhs[1:m.nz] = -differentiate(b, m.z).*tan(m.θ)

    # boundary conditions require zeros on RHS
    rhs[[1, 2, m.nz-1, m.nz]] .= 0

    # if dx(p) ~ 0 then 
    #   (1) set U for transport-constrained 1D solution
    #   (2) dz(nu*dzz(χ)) = b*tan(θ) at bottom
    #       for canonical 1D solution
    if m.transportConstraint
        rhs[iU] = m.U[1]
    else
        rhs[iU] = -b[1]*tan(m.θ) 
    end
    
    return rhs
end

"""
    χ, u, v, U = postProcess(m, sol)

Take solution `sol` and extract reshaped `χ`. Compute `u`, `v`
from definition of χ.
"""
function postProcess(m, sol)
    iU = m.nz + 1

    # transport at iU
    U = sol[iU] 

    # rest of solution is χ
    χ = sol[1:m.nz]

    # compute u = dz(χ)
    u = differentiate(χ, m.z)

    # compute v = int_-H^0 f*(χ - U)/nu dz
    v = cumtrapz(m.f*(χ .- U)./m.ν, m.z)

    return χ, u, v
end

"""
    χ, u, v = invert(m, b; bl=bl)

Wrapper function that inverts for flow given buoyancy perturbation `b`.
"""
function invert(m::ModelSetup1DPG, b::Array{Float64,1}; bl=false)
    if bl # BL Solution
        bz = differentiate(b, m.z)
        sol = @. -m.ν/m.f^2*bz*tan(m.θ)
        push!(sol, sol[end])
    else # full solution
        # compute RHS
        rhs = getInversionRHS(m, b)

        # solve full inversion
        sol = m.inversionLHS\rhs
    end

    # compute flow from sol
    χ, u, v = postProcess(m, sol)

    return χ, u, v
end
function invert!(m::ModelSetup1DPG, s::ModelState1DPG; bl=false)
    χ, u, v = invert(m, s.b; bl)
    s.χ[:] = χ
    s.u[:] = u
    s.v[:] = v
end
