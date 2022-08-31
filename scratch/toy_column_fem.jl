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
    return [poly[:contains_point](p[ip,:]) for ip=1:size(p, 1)]
end
function remove_outside_tris(p, t, pv)
    pmid = dropdims(sum(p[t,:], dims=2), dims=2) / 3
    is_inside = inpolygon(pmid, pv)
    return t[is_inside,:]
end

function lerp(x, x₁, y₁, x₂, y₂)
    return y₁*(x - x₂)/(x₁ - x₂) + y₂*(x - x₁)/(x₂ - x₁)
end

function get_grid(nξ::IT, nz::IT, L::FT, H₀::FT) where {FT <: Real, IT <: Integer}
    # horizontal gridpoints
    ξ = -L:2L/(nξ - 1):L

    # depth
    Δ = L/5
    G(x) = 1 - exp(-x^2/(2*Δ^2)) 
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    H = @. H₀*G(ξ - L)*G(ξ + L)

    # number of z points at each ξ
    dz = H₀/(nz - 1)
    nz = ones(Int64, nξ)
    for i=2:nξ-1
        nz[i] += Int64(ceil(H[i]/dz))
    end

    # generate grid
    p = zeros(Int64(sum(nz)), 2)
    println("np = ", size(p, 1))
    pv = zeros(nξ, 2)
    pv[1, :] = [-L 0]
    pv[nξ, :] = [L 0]
    e = zeros(Int64, 2*nξ - 2)
    c = 1
    p[1, :] = [L 0]
    e[1] = 1
    p[end, :] = [-L 0]
    e[end] = size(p, 1)
    for i=2:nξ-1
        n = nz[i]
        e[2*i - 2] = c + 1
        e[2*i - 1] = c + n
        pv[i, :] = [ξ[i] -H[i]]
        for j=1:n
            p[c + j, 1] = ξ[i]
            p[c + j, 2] = -H[i]*(cos(π*(j-1)/(nz[i]-1)) + 1)/2
        end
        c += n
    end

    # initial delaunay triangulation
    t = delaunay(p)

    # remove triangles outside the edges
    t = remove_outside_tris(p, t, pv)

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)

    # t_dict
    t_dict = get_t_dict(p, t)

    return ξ, H, p, t, e, C₀, t_dict
end
function get_grid(L::FT, H₀::FT) where FT <: Real
    p, t, e = load_mesh("../meshes/bowl1.h5")
    p[:, 1] *= L
    p[:, 2] *= H₀
    C₀ = get_shape_func_coeffs(p, t)
    t_dict = get_t_dict(p, t)
    return p, t, e, C₀, t_dict
end

"""
    A, b = get_A_b(p, t, e, C₀, δ)

Get matrix A and vector b to solve linear system representing the problem
    δ² u_zz + u = 1
in a finite element basis.
"""
function get_A_b(p::AbstractMatrix{FT}, t::AbstractMatrix{IT}, e::AbstractVector{IT},
                 C₀::AbstractArray{FT,3}, δ::FT) where {FT <: Real, IT <: Integer}
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
                func(ξ, η) = δ^2*shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dη=1)
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

    return A, b
end

nξ = 2^8
nz = 2^5
L = 5e6
H₀ = 2e3
# ξ, H, p, t, e, C₀, t_dict = get_grid(nξ, nz, L, H₀)
# p, t, e, C₀, t_dict = get_grid(L, H₀)
# fig, ax, im = tplot(p/1e3, t)
# # ax.scatter(p[e, 1]/1e3, p[e, 2]/1e3, 0.5)
# ax.set_xlabel(L"Zonal coordinate $x$ (km)")
# ax.set_ylabel(L"Vertical coordinate $z$ (km)")
# savefig("images/mesh.png")
# println("images/mesh.png")
# plt.close()

# δ = 20.
# A, b = get_A_b(p, t, e, C₀, δ)
# u = A\b
# fig, ax, im = tplot(p/1e3, t, u)
# cb = colorbar(im, ax=ax, label=L"$u$")
# ax.set_xlabel(L"Zonal coordinate $x$ (km)")
# ax.set_ylabel(L"Vertical coordinate $z$ (km)")
# savefig("images/u.png")
# println("images/u.png")
# plt.close()

# ξ₀ = 3e6
# bot_e = p[e, :]
# bot_e = bot_e[bot_e[:, 2] .< -1e-4, :]
# ξ = bot_e[:, 1]
# H = -bot_e[:, 2]
# i₁ = argmin(abs.(ξ .- ξ₀))
# if (ξ[i₁] > ξ₀) i₁ -= 1 end
# ξ₁ = ξ[i₁]
# H₁ = H[i₁]
# ξ₂ = ξ[i₁ + 1]
# H₂ = H[i₁ + 1]
# H = lerp(ξ₀, ξ₁, H₁, ξ₂, H₂)
# nz = 2^7
# z = @. -H*(cos(π*((1:nz)-1)/(nz-1)) + 1)/2
# u_profile = zeros(nz)
# for j=2:nz-1
#     # u_profile[j] = fem_evaluate(u, ξ₀, z[j], p, t, t_dict, C₀)
#     u_profile[j] = fem_evaluate(u, ξ₀, z[j], p, t, C₀)
# end
# u_exact = @. 1 - exp(-(z + H)/δ) - exp(z/δ)
# fig, ax = subplots(figsize=(1.955, 3.167))
# ax.plot(u_profile, z/1e3)
# ax.plot(u_exact, z/1e3, "k--", lw=0.5)
# ax.legend(["Numerical", "Exact"])
# ax.set_xlim(0, 1.1)
# # ax.set_ylim(-H/1e3, -(H + 200)/1e3)
# ax.set_xlabel(L"$u$")
# ax.set_ylabel(L"Vertical coordinate $z$ (km)")
# savefig("images/u_profile.png")
# println("images/u_profile.png")
# plt.close()

Δ = L/5
G(x) = 1 - exp(-x^2/(2*Δ^2)) 
H = @. H₀*G(p[:, 1] - L)*G(p[:, 1] + L)
u_exact = @. 1 - exp(-(p[:, 2] + H)/δ) - exp(p[:, 2]/δ)
abs_err = abs.(u - u_exact)
abs_err[e] .= 0
println(maximum(abs_err))