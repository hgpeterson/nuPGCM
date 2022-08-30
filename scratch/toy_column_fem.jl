using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    t = delaunay(p)

Delaunay triangulation `t` of N x 2 node array `p`.
"""
function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri[:Triangulation](p[:,1], p[:,2])
    return Int64.(t[:triangles] .+ 1)
end
"""
    inside = inpolygon(p, pv)

Determine if each point in the N x 2 node array `p` is inside the polygon
described by the NE x 2 node array `pv`.
"""
function inpolygon(p::AbstractMatrix{FT}, pv::AbstractMatrix{FT}) where FT <: Real
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    return [poly[:contains_point](p[ip,:]) for ip in eachindex(p)]
end
function remove_outside_tris(p, t, pv)
    pmid = dropdims(sum(p[t,:], dims=2), dims=2) / 3
    is_inside = inpolygon(pmid, pv)
    return t[is_inside,:]
end

function get_grid()
    L = 5e6
    nξ = 2^7
    ξ = -L:2L/(nξ - 1):L
    Δ = L/5
    G(x) = 1 - exp(-x^2/(2*Δ^2)) 
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    H₀ = 2e3
    H  = @. H₀*G(ξ - L)*G(ξ + L)
    Hx = @. H₀*(Gx(ξ - L)*G(ξ + L) + G(ξ - L)*Gx(ξ + L))

    dz = H₀/(2^6 - 1)
    nz = ones(Int64, nξ)
    for i=2:nξ-1
        nz[i] += Int64(ceil(H[i]/dz))
    end

    p = zeros(Int64(sum(nz)), 2)
    println("np = ", size(p, 1))
    e = zeros(Int64, nξ)
    c = 1
    p[1, :] = [ξ[1]   0]
    e[1] = 1
    p[end, :] = [ξ[end] 0]
    e[end] = size(p, 1)
    for i=2:nξ-1
        e[i] = c + 1
        n = nz[i]
        for j=1:n
            p[c + j, 1] = ξ[i]
            p[c + j, 2] = -H[i]*(cos(π*(j-1)/(nz[i]-1)) + 1)/2
        end
        c += n
    end

    # initial delaunay triangulation
    t = delaunay(p)

    # remove triangles outside the edges
    t = remove_outside_tris(p, t, p[e, :])

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)
    return p, t, e, C₀
end

function get_A_b(p::AbstractMatrix{FT}, t::AbstractMatrix{IT}, e::AbstractVector{IT},
                 C₀::AbstractArray{FT,3}) where {FT <: Real, IT <: Integer}
    np = size(p, 1)
    nt = size(t, 1)
    n = size(t, 2)
    A = Tuple{IT,IT,FT}[]
    b = zeros(FT, np)
    for k=1:nt        
        # calculate contribution to K from element k
        Kᵏ = zeros(FT, n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dη=1)
                Kᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to M from element k
        Mᵏ = zeros(FT, n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η)*shape_func(C₀[k, i, :], ξ, η)
                Mᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to b from element k
        bᵏ = zeros(FT, n)
        for i=1:n
            func(ξ, η) = 1*shape_func(C₀[k, i, :], ξ, η)
            bᵏ[i] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
        end

        # add to global system
        for i=1:n
            for j=1:n
                if t[k, i] in e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(A, (t[k, i], t[k, j], Kᵏ[i, j]))
                push!(A, (t[k, i], t[k, j], Mᵏ[i, j]))
            end
            b[t[k, i]] += bᵏ[i]
        end
    end
    # dirichlet u = 0 along edges
    for i in e
        push!(A, (i, i, 1))
    end
    b[e] .= 0

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), np, np)

    return lu(A), b
end

p, t, e, C₀ = get_grid()
# tplot(p, t)
# # scatter(p[:, 1], p[:, 2], 0.1)
# scatter(p[e, 1], p[e, 2], 1)
# savefig("images/debug.png")
# plt.close()

A, b = get_A_b(p, t, e, C₀)
u = A\b
fig, ax, im = tplot(p/1e3, t, u)
cb = colorbar(im, ax=ax, label=L"$u$")
ax.set_xlabel(L"Zonal coordinate $x$ (km)")
ax.set_ylabel(L"Vertical coordinate $z$ (km)")
savefig("images/u.png")
plt.close()