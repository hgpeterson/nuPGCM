"""
    D = get_D()

Compute the diffusion matrix needed for evolution equation integration.
"""
function get_D(z, κ)
    nz = length(z)
    D = Tuple{Int64,Int64,Float64}[] 

    # interior nodes 
    for j=2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        κ_z = sum(fd_z.*κ[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # diffusion term: dz(κ(N^2*cos(θ) + dz(b))) = dz(κ)*N^2*cos(θ) + dz(κ)*dz(b) + κ*dzz(b)
        push!(D, (j, j-1, (κ_z*fd_z[1] + κ[j]*fd_zz[1])))
        push!(D, (j, j,   (κ_z*fd_z[2] + κ[j]*fd_zz[2])))
        push!(D, (j, j+1, (κ_z*fd_z[3] + κ[j]*fd_zz[3])))
    end

    # flux at boundaries: bottom
    # dz stencil
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    # flux term: dz(b) = -N^2*cos(θ)
    push!(D, (1, 1, fd_z[1]))
    push!(D, (1, 2, fd_z[2]))
    push!(D, (1, 3, fd_z[3]))

    # flux at boundaries: top
    # dz stencil
    fd_z = mkfdstencil(z[nz-2:nz], z[nz], 1)
    # flux term: dz(b) = 0
    push!(D, (nz, nz-2, fd_z[1]))
    push!(D, (nz, nz-1, fd_z[2]))
    push!(D, (nz, nz,   fd_z[3]))

    # Create CSC sparse matrix from matrix elements
    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), nz, nz)

    return D
end