################################################################################
# Utility functions 
################################################################################

"""
    fẑ = ẑDerivative(field)

Compute dẑ(field) over 2D domain.
"""
function ẑDerivative(field)
    # allocate
    fẑ = zeros(nx, nz)

    # dẑ(field)
    for i=1:nx
        fẑ[i, :] .+= differentiate(field[i, :], ẑẑ[i, :])
    end

    return fẑ
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
# Inversion functions 
################################################################################

"""
    LHS = getInversionLHS()

Setup left hand side of linear system for problem.
"""
function getInversionLHS()
    nPts = nx*nz

    umap = reshape(1:nPts, nx, nz)    
    A = Tuple{Int64,Int64,Float64}[]  

    # Main loop, insert stencil in matrix for each node point
    for i=1:nx
        # for finite difference on the top and bottom boundary
        fd_bot_ẑ =  mkfdstencil(ẑẑ[i, 1:3],     ẑẑ[i, 1],  1)
        #= fd_top_ẑ =  mkfdstencil(ẑẑ[i, nz-2:nz], ẑẑ[i, nz], 1) =#
        fd_top_ẑẑ = mkfdstencil(ẑẑ[i, nz-3:nz], ẑẑ[i, nz], 2)

        # Lower boundary conditions 
        # b.c. 1: dẑ(chi) = 0
        push!(A, (umap[i, 1], umap[i, 1], fd_bot_ẑ[1]))
        push!(A, (umap[i, 1], umap[i, 2], fd_bot_ẑ[2]))
        push!(A, (umap[i, 1], umap[i, 3], fd_bot_ẑ[3]))
        # b.c. 2: chi = 0 
        push!(A, (umap[i, 2], umap[i, 1], 1.0))

        # Upper boundary conditions
        # b.c. 1: dẑẑ(chi) = 0 (or -stress at top)
        push!(A, (umap[i, nz], umap[i, nz-3], fd_top_ẑẑ[1]))
        push!(A, (umap[i, nz], umap[i, nz-2], fd_top_ẑẑ[2]))
        push!(A, (umap[i, nz], umap[i, nz-1], fd_top_ẑẑ[3]))
        push!(A, (umap[i, nz], umap[i, nz],   fd_top_ẑẑ[4]))
        # b.c. 2: chi = U = 0
        push!(A, (umap[i, nz-1], umap[i, nz], 1.0))

        # Interior nodes
        for j=3:nz-2
            row = umap[i, j] 

            # dẑ stencil
            fd_ẑ = mkfdstencil(ẑẑ[i, j-1:j+1], ẑẑ[i, j], 1)
            κ_ẑ = sum(fd_ẑ.*κ[i, j-1:j+1])

            # dẑẑ stencil
            fd_ẑẑ = mkfdstencil(ẑẑ[i, j-1:j+1], ẑẑ[i, j], 2)
            κ_ẑẑ = sum(fd_ẑẑ.*κ[i, j-1:j+1])

            # dẑẑẑ stencil
            fd_ẑẑẑ = mkfdstencil(ẑẑ[i, j-2:j+2], ẑẑ[i, j], 3)

            # dẑẑẑẑ stencil
            fd_ẑẑẑẑ = mkfdstencil(ẑẑ[i, j-2:j+2], ẑẑ[i, j], 4)
            
            # eqtn: dẑẑ(nu*dẑẑ(chi)) + f^2*cos^2(θ)(chi - U)/nu = -dẑ(b)*sin(θ)
            # term 1 (product rule)
            push!(A, (row, umap[i, j-1], Pr*κ_ẑẑ*fd_ẑẑ[1]))
            push!(A, (row, umap[i, j],   Pr*κ_ẑẑ*fd_ẑẑ[2]))
            push!(A, (row, umap[i, j+1], Pr*κ_ẑẑ*fd_ẑẑ[3]))

            push!(A, (row, umap[i, j-2], 2*Pr*κ_ẑ*fd_ẑẑẑ[1]))
            push!(A, (row, umap[i, j-1], 2*Pr*κ_ẑ*fd_ẑẑẑ[2]))
            push!(A, (row, umap[i, j],   2*Pr*κ_ẑ*fd_ẑẑẑ[3]))
            push!(A, (row, umap[i, j+1], 2*Pr*κ_ẑ*fd_ẑẑẑ[4]))
            push!(A, (row, umap[i, j+2], 2*Pr*κ_ẑ*fd_ẑẑẑ[5]))

            push!(A, (row, umap[i, j-2], Pr*κ[i, j]*fd_ẑẑẑẑ[1]))
            push!(A, (row, umap[i, j-1], Pr*κ[i, j]*fd_ẑẑẑẑ[2]))
            push!(A, (row, umap[i, j],   Pr*κ[i, j]*fd_ẑẑẑẑ[3]))
            push!(A, (row, umap[i, j+1], Pr*κ[i, j]*fd_ẑẑẑẑ[4]))
            push!(A, (row, umap[i, j+2], Pr*κ[i, j]*fd_ẑẑẑẑ[5]))
            # term 2
            push!(A, (row, umap[i, j], f^2*cosθ[i, j]^2/(Pr*κ[i, j])))
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nPts, nPts)

    return A
end

"""
    RHS = getInversionRHS(b)

Setup right hand side of linear system for problem.
"""
function getInversionRHS(b)
    nPts = nx*nz
    
    # eqtn: dẑẑ(nu*dẑẑ(chi)) + f^2*cos^2(θ)(chi - U)/nu = -dẑ(b)*sin(θ)
    rhs = -ẑDerivative(b).*sinθ

    # boundary conditions require zeros on RHS
    rhs[:, [1, 2, nz-1, nz]] .= 0

    # reshape to vector
    rhs = reshape(rhs, nPts, 1)
    
    return rhs
end

"""
    chi, û, v = postProcess(sol)

Take solution `sol` and extract reshaped `chi`. Compute `û`, `v`
from definition of chi.
"""
function postProcess(sol)
    # reshape 
    chi = reshape(sol, nx, nz)

    # compute û = dẑ(chi)
    û = ẑDerivative(chi)

    # compute v = int_-H^0 f*cos(θ)*chi/nu dẑ
    v = zeros(nx, nz)
    for i=1:nx
        v[i, :] = cumtrapz(f*cosθ[i, 1]*chi[i, :]./(Pr*κ[i, :]), ẑẑ[i, :])
    end

    return chi, û, v
end

"""
    chi, û, v,  = invert(b, inversionLHS)

Wrapper function that inverts for flow given buoyancy perturbation `b`.
"""
function invert(b, inversionLHS)
    # compute RHS
    inversionRHS = getInversionRHS(b)

    # solve
    sol = inversionLHS\inversionRHS

    # compute flow from sol
    chi, û, v = postProcess(sol)

    return chi, û, v
end

################################################################################
