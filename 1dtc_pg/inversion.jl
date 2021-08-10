"""
    LHS = getInversionLHS()

Setup left hand side of linear system for problem.
"""
function getInversionLHS()
    iU = nẑ + 1
    A = Tuple{Int64,Int64,Float64}[]  

    # for finite difference on the top and bottom boundary
    fd_bot_ẑ =  mkfdstencil(ẑ[1:3], ẑ[1],  1)
    fd_top_ẑẑ = mkfdstencil(ẑ[nẑ-3:nẑ], ẑ[nẑ], 2)

    # Lower boundary conditions 
    # b.c. 1: dẑ(χ) = 0
    push!(A, (1, 1, fd_bot_ẑ[1]))
    push!(A, (1, 2, fd_bot_ẑ[2]))
    push!(A, (1, 3, fd_bot_ẑ[3]))
    # b.c. 2: χ = 0 
    push!(A, (2, 1, 1.0))

    # Upper boundary conditions
    # b.c. 1: dẑẑ(χ) = 0 
    push!(A, (nẑ, nẑ-3, fd_top_ẑẑ[1]))
    push!(A, (nẑ, nẑ-2, fd_top_ẑẑ[2]))
    push!(A, (nẑ, nẑ-1, fd_top_ẑẑ[3]))
    push!(A, (nẑ, nẑ,   fd_top_ẑẑ[4]))
    # b.c. 2: χ - U = 0
    push!(A, (nẑ-1, nẑ,  1.0))
    push!(A, (nẑ-1, iU, -1.0))

    # Interior nodes
    for j=3:nẑ-2
        row = j 

        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 1)
        κ_ẑ = sum(fd_ẑ.*κ[j-1:j+1])

        # dẑẑ stencil
        fd_ẑẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 2)
        κ_ẑẑ = sum(fd_ẑẑ.*κ[j-1:j+1])

        # dẑẑẑ stencil
        fd_ẑẑẑ = mkfdstencil(ẑ[j-2:j+2], ẑ[j], 3)

        # dẑẑẑẑ stencil
        fd_ẑẑẑẑ = mkfdstencil(ẑ[j-2:j+2], ẑ[j], 4)
        
        # eqtn: dẑẑ(nu*dẑẑ(χ)) + f^2*cos^2(θ)(χ - U)/nu = -dẑ(b)*sin(θ)
        # term 1 (product rule)
        push!(A, (row, j-1, Pr*κ_ẑẑ*fd_ẑẑ[1]))
        push!(A, (row, j,   Pr*κ_ẑẑ*fd_ẑẑ[2]))
        push!(A, (row, j+1, Pr*κ_ẑẑ*fd_ẑẑ[3]))

        push!(A, (row, j-2, 2*Pr*κ_ẑ*fd_ẑẑẑ[1]))
        push!(A, (row, j-1, 2*Pr*κ_ẑ*fd_ẑẑẑ[2]))
        push!(A, (row, j,   2*Pr*κ_ẑ*fd_ẑẑẑ[3]))
        push!(A, (row, j+1, 2*Pr*κ_ẑ*fd_ẑẑẑ[4]))
        push!(A, (row, j+2, 2*Pr*κ_ẑ*fd_ẑẑẑ[5]))

        push!(A, (row, j-2, Pr*κ[j]*fd_ẑẑẑẑ[1]))
        push!(A, (row, j-1, Pr*κ[j]*fd_ẑẑẑẑ[2]))
        push!(A, (row, j,   Pr*κ[j]*fd_ẑẑẑẑ[3]))
        push!(A, (row, j+1, Pr*κ[j]*fd_ẑẑẑẑ[4]))
        push!(A, (row, j+2, Pr*κ[j]*fd_ẑẑẑẑ[5]))
        # term 2
        push!(A, (row, j,   f^2*cos(θ)^2/(Pr*κ[j])))
        push!(A, (row, iU, -f^2*cos(θ)^2/(Pr*κ[j])))
    end

    # if dx(p) ~ 0 then 
    #   (1) U = U₀
    #       for transport-constrained 1D solution
    #   (2) dẑ(nu*dẑẑ(χ)) = Hx*b at bottom
    #       for canonical 1D solution
    row = iU
    if transportConstraint
        push!(A, (row, row, 1.0))
    else
        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑ[1:3], ẑ[1], 1)
        κ_ẑ = sum(fd_ẑ.*κ[1:3])

        # dẑẑ stencil
        fd_ẑẑ = mkfdstencil(ẑ[1:4], ẑ[1], 2)

        # dẑẑẑ stencil
        fd_ẑẑẑ = mkfdstencil(ẑ[1:5], ẑ[1], 3)

        # product rule
        push!(A, (row, 1, Pr*κ_ẑ*fd_ẑẑ[1]))
        push!(A, (row, 2, Pr*κ_ẑ*fd_ẑẑ[2]))
        push!(A, (row, 3, Pr*κ_ẑ*fd_ẑẑ[3]))
        push!(A, (row, 4, Pr*κ_ẑ*fd_ẑẑ[4]))

        push!(A, (row, 1, Pr*κ[1]*fd_ẑẑẑ[1]))
        push!(A, (row, 2, Pr*κ[1]*fd_ẑẑẑ[2]))
        push!(A, (row, 3, Pr*κ[1]*fd_ẑẑẑ[3]))
        push!(A, (row, 4, Pr*κ[1]*fd_ẑẑẑ[4]))
        push!(A, (row, 5, Pr*κ[1]*fd_ẑẑẑ[5]))
    end
    
    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nẑ+1, nẑ+1)

    return A
end

"""
    RHS = getInversionRHS(b)

Setup right hand side of linear system for problem.
"""
function getInversionRHS(b)
    # last row is for U
    rhs = zeros(nẑ+1)
    iU = nẑ + 1

    # eqtn: dẑẑ(nu*dẑẑ(χ)) + f^2*cos^2(θ)(χ - U)/nu = -dẑ(b)*sin(θ)
    rhs[1:nẑ] = -differentiate(b, ẑ).*sin(θ)

    # boundary conditions require zeros on RHS
    rhs[[1, 2, nẑ-1, nẑ]] .= 0

    # if dx(p) ~ 0 then 
    #   (1) U = U₀
    #       for transport-constrained 1D solution
    #   (2) dẑ(nu*dẑẑ(χ)) = Hx*b at bottom
    #       for canonical 1D solution
    if transportConstraint
        rhs[iU] = U₀
    else
        rhs[iU] = -b[1]*tan(θ) #FIXME: Why isn't this sin(θ)?
    end
    
    return rhs
end

"""
    χ, û, v, U = postProcess(sol)

Take solution `sol` and extract reshaped `χ`. Compute `û`, `v`
from definition of χ.
"""
function postProcess(sol)
    iU = nẑ + 1

    # χ at top is vertical integral of û
    U = sol[iU] 

    # rest of solution is χ
    χ = sol[1:nẑ]

    # compute û = dẑ(χ)
    û = differentiate(χ, ẑ)

    # compute v̂ = int_-H^0 f*cos(θ)*(χ - U)/nu dẑ
    v̂ = cumtrapz(f*cos(θ)*(χ .- U)./(Pr*κ), ẑ)

    return χ, û, v̂, U
end

"""
    χ, û, v̂, U = invert(b)

Wrapper function that inverts for flow given buoyancy perturbation `b`.
"""
function invert(b)
    # compute RHS
    inversionRHS = getInversionRHS(b)

    # solve
    inversionRHS = getInversionRHS(b)
    sol = inversionLHS\inversionRHS

    # compute flow from sol
    χ, û, v̂, U = postProcess(sol)

    return χ, û, v̂, U
end
