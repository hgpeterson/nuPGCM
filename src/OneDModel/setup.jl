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
    K, v = build_diffusion_system(z, κ, N²)

Build matrix/vector representation of 
    dz(κ*(N² + dz(b))) 
        = dz(κ*dz(b)) + N²dz(κ) 
        = K*b + rhs_diff
The matrix K also contains the boundary conditions 
    b = 0 at z = 0
    N² + dz(b) = 0 at z = -H
for the first and last rows, respectively.
"""
function build_diffusion_system(z, κ, N²)
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
        push!(K, (j, j-1, (κ_z*fd_z[1] + κ[j]*fd_zz[1])))
        push!(K, (j, j,   (κ_z*fd_z[2] + κ[j]*fd_zz[2])))
        push!(K, (j, j+1, (κ_z*fd_z[3] + κ[j]*fd_zz[3])))

        # N²*dz(κ) has no dependence on b -> vector
        v[j] += N²*κ_z
    end

    # z = -H: N² + dz(b) = 0 -> dz(b) = -N²
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    push!(K, (1, 1, fd_z[1]))
    push!(K, (1, 2, fd_z[2]))
    push!(K, (1, 3, fd_z[3]))

    # z = 0: b = 0
    push!(K, (nz, nz, 1))

    # Create CSC sparse matrices
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nz, nz)

    return K, v
end

"""
Build matrix representation of
   -α²ε²sec²θ dz(ν*dz(u)) - f*v + Px = b*tan(θ)/α
   -α²ε²sec²θ dz(ν*dz(v)) + f*u + Py = 0
Boundary conditions:
    dz(u) = dz(v) = 0 at z = 0
    u = v = 0 at z = -H
    ∫ u dz = 0 or Px = 0 depending on params.no_Px
    ∫ v dz = 0 or Py = 0 depending on params.no_Py
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
        
        # eq 1: -α²ε²sec²θ dz(ν*dz(u)) - f*v + Px = b*tan(θ)/α
        # term 1 = -α²ε²sec²θ [dz(ν)*dz(u) + ν*dzz(u)] 
        c = α^2*ε^2*sec(θ)^2
        push!(LHS, (umap[j], umap[j-1], -c*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (umap[j], umap[j],   -c*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (umap[j], umap[j+1], -c*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = -f*v
        push!(LHS, (umap[j], vmap[j], -f))
        # term 3 = Px
        push!(LHS, (umap[j], iPx, 1))

        # eq 2: -α²ε²sec²θ dz(ν*dz(v)) + f*u + Py = 0
        # term 1 = -α²ε²sec²θ [dz(ν)*dz(v) + ν*dzz(v)]
        push!(LHS, (vmap[j], vmap[j-1], -c*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (vmap[j], vmap[j],   -c*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (vmap[j], vmap[j+1], -c*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = f*u
        push!(LHS, (vmap[j], umap[j], f))
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

    # ∫ u dz = 0 or Px = 0
    if params.no_Px
        push!(LHS, (iPx, iPx, 1))
    else
        for j in 1:nz-1
            # trapezoidal rule
            dz = z[j+1] - z[j]
            push!(LHS, (iPx, umap[j],   dz/2))
            push!(LHS, (iPx, umap[j+1], dz/2))
        end
    end

    # ∫ v dz = 0 or Py = 0
    if params.no_Py
        push!(LHS, (iPy, iPy, 1))
    else
        for j in 1:nz-1
            # trapezoidal rule
            dz = z[j+1] - z[j]
            push!(LHS, (iPy, vmap[j],   dz/2))
            push!(LHS, (iPy, vmap[j+1], dz/2))
        end
    end

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), 2nz+2, 2nz+2)

    return LHS
end

"""
Update vector for RHS of inversion 
   -α²ε²sec²θ dz(ν*dz(u)) - f*v + Px = b*tan(θ)/α
   -α²ε²sec²θ dz(ν*dz(v)) + f*u + Py = 0
"""
function update_rhs_inversion!(rhs, b, params)
    rhs[2:params.nz-1] .= b[2:params.nz-1]*tan(params.θ)/params.α
    return rhs
end