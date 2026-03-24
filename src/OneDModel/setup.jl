"""
    x = chebyshev_nodes(n)

Return `n` Chebyshev nodes in the interval `[-1, 0]`.
"""
function chebyshev_nodes(n)
    return ([-cos((i - 1)*π/(n - 1)) for i ∈ 1:n] .- 1)/2
end

"""
    LHS = assemble_LHS_evolution(K, a)

Assemble the LHS of evolution system given diffusion matrix `K` (which holds b.c. information on first and last rows)
and BDF parameter `a`.
"""
function assemble_LHS_evolution(K, a)
    # BDF scheme
    LHS = I - a*K

    # reset boundary conditions
    LHS[1, :] = K[1, :]
    LHS[end, :] = K[end, :]

    return LHS
end

"""
    K, v = build_diffusion_system(z, κ, N², θ)

Build matrix/vector representation of 
    dz(κ*(N²cos(θ) + dz(b))) 
        = dz(κ*dz(b)) + N²dz(κ)cos(θ)
        = K*b + rhs_diff
The matrix K also contains the boundary conditions 
    b = 0 at z = 0
    N²cos(θ) + dz(b) = 0 at z = -H
for the first and last rows, respectively.
"""
function build_diffusion_system(z, κ, N², θ)
    # initialize
    nz = length(z)
    K = Tuple{Int64,Int64,Float64}[] 
    v = zeros(nz)

    # interior nodes 
    for j=2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)

        # dz(κ)
        κ_z = fd_z[1]*κ[j-1] + fd_z[2]*κ[j] + fd_z[3]*κ[j+1]

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # product rule: dz(κ*dz(b)) = dz(κ)*dz(b) + κ*dzz(b)
        push!(K, (j, j-1, κ_z*fd_z[1] + κ[j]*fd_zz[1]))
        push!(K, (j, j,   κ_z*fd_z[2] + κ[j]*fd_zz[2]))
        push!(K, (j, j+1, κ_z*fd_z[3] + κ[j]*fd_zz[3]))

        # N²*dz(κ)*cos(θ) has no dependence on b -> vector
        v[j] += N²*κ_z*cos(θ)
    end

    # z = -H: N²cos(θ) + dz(b) = 0 -> dz(b) = -N²cosθ
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    push!(K, (1, 1, fd_z[1]))
    push!(K, (1, 2, fd_z[2]))
    push!(K, (1, 3, fd_z[3]))

    # # z = 0: b = 0
    # push!(K, (nz, nz, 1))

    # z = 0: dz(b) = 0
    fd_z = mkfdstencil(z[nz-2:nz], z[nz], 1)
    push!(K, (nz, nz-2, fd_z[1]))
    push!(K, (nz, nz-1, fd_z[2]))
    push!(K, (nz, nz,   fd_z[3]))

    # Create CSC sparse matrices
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nz, nz)

    return K, v
end

"""
Build matrix representation of
   -α²ε²dz(ν*dz(u)) - f*v*cos(θ) + Px = b*sin(θ)/α
   -α²ε²dz(ν*dz(v)) + f*u*cos(θ) + Py = 0
Boundary conditions:
    dz(u) = dz(v) = 0 at z = 0
    u = v = 0 at z = -H
    ∫ u dz = params.U or Px = params.Px
    ∫ v dz = params.V or Py = params.Py
"""
function build_LHS_inversion(z, ν, params)
    # unpack
    α = params.α
    ε = params.ε
    θ = params.θ
    f = params.f

    # setup
    nz = length(z)
    umap = 1:nz
    vmap = nz+1:2nz
    iPx = 2nz + 1
    iPy = 2nz + 2
    LHS = Tuple{Int64,Int64,Float64}[]  

    # interior nodes
    for j ∈ 2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)
        
        # eq 1: -α²ε²dz(ν*dz(u)) - f*v*cos(θ) + Px = b*sin(θ)/α
        # term 1 = -α²ε²[dz(ν)*dz(u) + ν*dzz(u)] 
        push!(LHS, (umap[j], umap[j-1], -α^2*ε^2*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (umap[j], umap[j],   -α^2*ε^2*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (umap[j], umap[j+1], -α^2*ε^2*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = -f*v*cos(θ)
        push!(LHS, (umap[j], vmap[j], -f*cos(θ)))
        # term 3 = Px
        push!(LHS, (umap[j], iPx, 1))

        # eq 2: -α²ε²dz(ν*dz(v)) + f*u*cos(θ) + Py = 0
        # term 1 = -α²ε²[dz(ν)*dz(v) + ν*dzz(v)]
        push!(LHS, (vmap[j], vmap[j-1], -α^2*ε^2*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (vmap[j], vmap[j],   -α^2*ε^2*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (vmap[j], vmap[j+1], -α^2*ε^2*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = f*u*cos(θ)
        push!(LHS, (vmap[j], umap[j], f*cos(θ)))
        # term 3 = Py
        push!(LHS, (vmap[j], iPy, 1))
    end

    # bottom boundary conditions: u = v = 0
    push!(LHS, (umap[1], umap[1], 1))
    push!(LHS, (vmap[1], vmap[1], 1))

    # surface boundary conditions: dz(u) = dz(v) = 0
    fd_z = mkfdstencil(z[end-2:end], z[end], 1)
    push!(LHS, (umap[end], umap[end-2], fd_z[1]))
    push!(LHS, (umap[end], umap[end-1], fd_z[2]))
    push!(LHS, (umap[end], umap[end],   fd_z[3]))
    push!(LHS, (vmap[end], vmap[end-2], fd_z[1]))
    push!(LHS, (vmap[end], vmap[end-1], fd_z[2]))
    push!(LHS, (vmap[end], vmap[end],   fd_z[3]))

    if !isnothing(params.Px)
        if !isnothing(params.U)
            throw(ArgumentError("Must set either Px or U, not both"))
        end
        # set Px = Px
        push!(LHS, (iPx, iPx, 1))
    elseif !isnothing(params.U)
        # set ∫ u dz = U
        for j in 1:nz-1
            # trapezoidal rule
            dz = z[j+1] - z[j]
            push!(LHS, (iPx, umap[j],   dz/2))
            push!(LHS, (iPx, umap[j+1], dz/2))
        end
    else
        throw(ArgumentError("Must set either Px or U"))
    end

    if !isnothing(params.Py)
        if !isnothing(params.V)
            throw(ArgumentError("Must set either Py or V, not both"))
        end
        # set Py = Py
        push!(LHS, (iPy, iPy, 1))
    elseif !isnothing(params.V)
        # set ∫ v dz = V
        for j in 1:nz-1
            # trapezoidal rule
            dz = z[j+1] - z[j]
            push!(LHS, (iPy, vmap[j],   dz/2))
            push!(LHS, (iPy, vmap[j+1], dz/2))
        end
    else
        throw(ArgumentError("Must set either Py or V"))
    end

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), 2nz+2, 2nz+2)

    return LHS
end

"""
Update vector for RHS of inversion 
   -α²ε²dz(ν*dz(u)) - f*v*cos(θ) + Px = b*sin(θ)/α
   -α²ε²dz(ν*dz(v)) + f*u*cos(θ) + Py = 0
"""
function update_rhs_inversion!(rhs, b, params)
    rhs[2:params.nz-1] .= b[2:params.nz-1]*sin(params.θ)/params.α
    if !isnothing(params.Px)
        rhs[2params.nz+1] = params.Px
    else  # error cases should be caught above
        rhs[2params.nz+1] = params.U
    end
    if !isnothing(params.Py)
        rhs[2params.nz+2] = params.Py
    else  # error cases should be caught above
        rhs[2params.nz+2] = params.V
    end
    return rhs
end