"""
    LHS = getInversionLHS()

Setup left hand side of linear system for problem.
"""
function getInversionLHS()
    iU = nẑ + 1
    A = Tuple{Int64,Int64,Float64}[]  

    # Lower boundary condition: χ = 0 
    push!(A, (1, 1, 1.0))

    # Upper boundary condition: χ - U = 0
    push!(A, (nẑ, nẑ,  1.0))
    push!(A, (nẑ, iU, -1.0))

    # Interior nodes
    for j=2:nẑ-1
        row = j 

        # dẑẑ stencil
        fd_ẑẑ = mkfdstencil(ẑ[j-1:j+1], ẑ[j], 2)
        
        # eqtn: (f^2 + r^2)/r * dẑẑ(χ)*cos(θ)^2 = dẑ(b)*sin(θ)
        push!(A, (row, j-1, (f^2 + r^2)/r * fd_ẑẑ[1] * cos(θ)^2))
        push!(A, (row, j,   (f^2 + r^2)/r * fd_ẑẑ[2] * cos(θ)^2))
        push!(A, (row, j+1, (f^2 + r^2)/r * fd_ẑẑ[3] * cos(θ)^2))
    end

    # if dx(p) ~ 0 then 
    #   (1) U = U₀
    #       for transport-constrained 1D solution
    #   (2) (f^2 + r^2)/r * dẑ(χ) * cos(θ)^2 = b*sin(θ) at bottom
    #       for canonical 1D solution
    row = iU
    if transportConstraint
        push!(A, (row, row, 1.0))
    else
        # dẑ stencil
        fd_ẑ = mkfdstencil(ẑ[1:3], ẑ[1], 1)

        push!(A, (row, 1, (f^2 + r^2)/r * fd_ẑ[1] * cos(θ)^2))
        push!(A, (row, 2, (f^2 + r^2)/r * fd_ẑ[2] * cos(θ)^2))
        push!(A, (row, 3, (f^2 + r^2)/r * fd_ẑ[3] * cos(θ)^2))
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

    # eqtn: (f^2 + r^2)/r * dẑẑ(χ)*cos(θ)^2 = dẑ(b)*sin(θ)
    rhs[1:nẑ] = differentiate(b, ẑ).*sin(θ)

    # boundary conditions require zeros on RHS
    rhs[[1, nẑ]] .= 0

    # if dx(p) ~ 0 then 
    #   (1) U = U₀
    #       for transport-constrained 1D solution
    #   (2) (f^2 + r^2)/r * dẑ(χ) * cos(θ)^2 = b*sin(θ) at bottom
    #       for canonical 1D solution
    if transportConstraint
        rhs[iU] = U₀
    else
        rhs[iU] = b[1]*sin(θ)
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

    # compute v̂ = -f*û*cos(θ)/r
    v̂ = -f*û*cos(θ)/r

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

"""
    b, u, v, w = pointwise1DConstantκ(t)

Apply the 1D solution to the Rayleigh drag problem pointwise over the domain.
See CF18 for details.
"""
function pointwise1DConstantκ(t)
    # inverse boundary layer thickness
    q = sqrt(r*N^2*tan(θ)^2/(κ[1]*(f^2 + r^2)))

    # time dependent analytical buoyancy solution (only works for constant κ)
    b = @. N^2*cos(θ)/q*(exp(-q*(ẑ + H)) - 0.5*(exp(-q*(ẑ + H))*erfc(q*sqrt(κ[1]*t) - (ẑ + H)/2/sqrt(κ[1]*t)) + exp(q*(ẑ + H))*erfc(q*sqrt(κ[1]*t) + (ẑ + H)/2/sqrt(κ[1]*t))))

    # invert for flow using rotated 1D equations
    û = @. b*sin(θ)/((f^2 + r^2)*cos(θ)^2/r)
    v = @. -f*û*cos(θ)/r

    # rotate
    u = @. û*cos(θ)
    w = @. û*sin(θ)

    return b, u, v, w
end