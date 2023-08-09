"""
    Dξ = get_Dξ(ξ, L, periodic)

Compute the 2D ξ derivative matrix.
"""
function get_Dξ(ξ, L, periodic)
    nξ = size(ξ, 1)
    
    Dξ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for i=1:nξ
        if i == 1
            # left 
            if periodic
                fd_ξ = mkfdstencil([ξ[nξ] - L, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (i, nξ, fd_ξ[1]))
                push!(Dξ, (i, 1,  fd_ξ[2]))
                push!(Dξ, (i, 2,  fd_ξ[3]))
            else
                # ghost point at ξ = 0 where derivative is zero:
                #   fd_ξ0[1]*f[0] = -fd_ξ0[2]*f[1] - fd_ξ0[3]*f[2]
                #   -> dξ(f) at ξ[1] = fd_ξ[1]*f[0] + fd_ξ[2]*f[1] + fd_ξ[3]*f[2]
                #                    = -fd_ξ[1]*(fd_ξ0[2]*f[1] + fd_ξ0[3]*f[2])/fd_ξ0[1] + fd_ξ[2]*f[1] + fd_ξ[3]*f[2]
                #                    = (fd_ξ[2] - fd_ξ[1]*fd_ξ0[2]/fd_ξ0[1]) * f[1] + (fd_ξ[3] - fd_ξ[1]fd_ξ0[3]/fd_ξ0[1]) * f[2]
                fd_ξ0 = mkfdstencil([0, ξ[1], ξ[2]], 0.0, 1) 
                fd_ξ = mkfdstencil([0, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (i, 1, fd_ξ[2] - fd_ξ[1]*fd_ξ0[2]/fd_ξ0[1]))
                push!(Dξ, (i, 2, fd_ξ[3] - fd_ξ[1]*fd_ξ0[3]/fd_ξ0[1]))

                # fd_ξ = mkfdstencil(ξ[1:3], ξ[1], 1) 
                # push!(Dξ, (i, 1, fd_ξ[1]))
                # push!(Dξ, (i, 2, fd_ξ[2]))
                # push!(Dξ, (i, 3, fd_ξ[3]))
            end
        elseif i == nξ
            # right
            if periodic
                fd_ξ = mkfdstencil([ξ[nξ-1], ξ[nξ], ξ[1] + L], ξ[nξ], 1)
                push!(Dξ, (i, nξ-1, fd_ξ[1]))
                push!(Dξ, (i, nξ,   fd_ξ[2]))
                push!(Dξ, (i, 1,    fd_ξ[3]))
            else
                fd_ξ = mkfdstencil(ξ[nξ-2:nξ], ξ[nξ], 1)
                push!(Dξ, (i, nξ-2, fd_ξ[1]))
                push!(Dξ, (i, nξ-1, fd_ξ[2]))
                push!(Dξ, (i, nξ,   fd_ξ[3]))
            end
        else
            # interior
            fd_ξ = mkfdstencil(ξ[i-1:i+1], ξ[i], 1)
            push!(Dξ, (i, i-1, fd_ξ[1]))
            push!(Dξ, (i, i,   fd_ξ[2]))
            push!(Dξ, (i, i+1, fd_ξ[3]))
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dξ = sparse((x->x[1]).(Dξ), (x->x[2]).(Dξ), (x->x[3]).(Dξ), nξ, nξ)

    return Dξ
end

"""
    Dσ = get_Dσ(σ)

Compute the σ derivative matrix.
"""
function get_Dσ(σ)
    nσ = size(σ, 1)

    Dσ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for j=1:nσ
        if j == 1 
            # bottom 
            fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
            push!(Dσ, (j, 1, fd_σ[1]))
            push!(Dσ, (j, 2, fd_σ[2]))
            push!(Dσ, (j, 3, fd_σ[3]))
        elseif j == nσ
            # top 
            fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
            push!(Dσ, (j, nσ-2, fd_σ[1]))
            push!(Dσ, (j, nσ-1, fd_σ[2]))
            push!(Dσ, (j, nσ,   fd_σ[3]))
        else
            # interior
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            push!(Dσ, (j, j-1, fd_σ[1]))
            push!(Dσ, (j, j,   fd_σ[2]))
            push!(Dσ, (j, j+1, fd_σ[3]))
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dσ = sparse((x->x[1]).(Dσ), (x->x[2]).(Dσ), (x->x[3]).(Dσ), nσ, nσ)

    return Dσ
end

"""
    D = get_D(ξ, σ, κ, H)

Compute diffusion matrix needed for 2D evolution
"""
function get_D(ξ, σ, κ,  H)
    nξ = size(ξ, 1)
    nσ = size(σ, 1)
    n_pts = nξ*nσ

    umap = reshape(1:n_pts, nξ, nσ)    
    D = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix (with boundary flux conditions)

    # Main loop, insert stencil in matrices for each node point
    for i=1:nξ
        # interior nodes only for operators
        for j=2:nσ-1
            row = umap[i, j] 

            # dσ stencil
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            κ_σ = sum(fd_σ.*κ[i, j-1:j+1])

            # dσσ stencil
            fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

            # diffusion term: dσ(κ*dσ(b))/H^2 = 1/H^2*(dσ(κ)*dσ(b) + κ*dσσ(b))
            push!(D, (row, umap[i, j-1], (κ_σ*fd_σ[1] + κ[i, j]*fd_σσ[1])/H[i]^2))
            push!(D, (row, umap[i, j],   (κ_σ*fd_σ[2] + κ[i, j]*fd_σσ[2])/H[i]^2))
            push!(D, (row, umap[i, j+1], (κ_σ*fd_σ[3] + κ[i, j]*fd_σσ[3])/H[i]^2))
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, 1], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, 2], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, 3], fd_σ[3]/H[i]))

        # flux at boundaries: top
        row = umap[i, nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, nσ-2], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, nσ-1], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, nσ],   fd_σ[3]/H[i]))
    end

    # Create CSC sparse matrix from matrix elements
    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), n_pts, n_pts)

    return D
end

"""
    dfdξ = ∂ξ(m, field)

Compute derivative of `field` in ξ-direction in terrian-following coordinates.
"""
function ∂ξ(m::ModelSetup2D, field)
    return m.Dξ*field
end

"""
    dfdσ = ∂σ(m, field)

Compute derivative of `field` in σ-direction in terrian-following coordinates.
"""
function ∂σ(m::ModelSetup2D, field)
    return (m.Dσ*field')'
end

"""
    dfdx = ∂x(m, field)

Compute derivative of `field` in x-direction in terrian-following coordinates.
Note: ∂x() = ∂ξ() - ∂x(H)*σ*∂σ()/H
"""
function ∂x(m::ModelSetup2D, field)
    # ∂ξ(field)
    dfdx = ∂ξ(m, field)

    # -∂x(H)*σ*∂σ(field)/H
    dfdx -= repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*∂σ(m, field)

    return dfdx
end

"""
    fz = ∂z(m, field)

Compute dz(`field`) in terrian-following coordinates.
Note: dz() = dσ()/H
"""
function ∂z(m::ModelSetup2D, field)
    # dσ(field)/H
    fz = ∂σ(m, field)./repeat(m.H, 1, m.nσ)
    return fz
end

"""
    u, v, w = transform_from_TF(m, s)

Transform from terrain-following coordinates to cartesian coordinates.
"""
function transform_from_TF(m::ModelSetup2D, s::ModelState2D)
    u = s.uξ
    v = s.uη
    w = s.uσ.*repeat(m.H, 1, m.nσ) + repeat(m.σ', m.nξ, 1).*repeat(m.Hx, 1, m.nσ).*s.uξ
    return u, v, w
end
