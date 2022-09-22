using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    u, p = solve_stokes()

Stokes problem:
    -Δu + ∇p = 0      on Ω,
         ∇⋅u = 0      on Ω,
           u = 0      on Γ₁,
           u = (1, 0) on Γ₂,
with extra condition
    ∫ p dx = 0.
Here u = (u₁, u₂) is the velocity vector and p is the pressure.
Weak form:
    ∫ (∇u)⊙(∇v) - p (∇⋅v) dx = 0,
    ∫ q (∇⋅u) dx = 0,
for all 
    v ∈ V = {(v₁, v₂) | vᵢ ∈ P₂},
    q ∈ Q = {q ∈ P₁ | ∫ q dx = 0},
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes(g₁, g₂, sfi_uu, sfi_pu, J, Γ₁, Γ₂) 
    # indices
    umap = reshape(1:2*g₂.np, (2, g₂.np))
    # umap = reshape(1:2*g₂.np, (g₂.np, 2))'
    pmap = umap[end] .+ (1:g₁.np)
    N = pmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)
    for k=1:g₁.nt
        # contribution from (∇u)⊙(∇v) term 
        Kᵏ = abs(J.J[k])*(sfi_uu.φξφξ*(J.ξx[k]^2       + J.ξy[k]^2) + 
                          sfi_uu.φξφη*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                          sfi_uu.φηφξ*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                          sfi_uu.φηφη*(J.ηx[k]^2       + J.ηy[k]^2))

        # contribution from p*(∇⋅v) term
        # Cᵏ = abs(J.J[k])*(sfi_pu.φφξ*(J.ξx[k] + J.ξy[k]) + sfi_pu.φφη*(J.ηx[k] + J.ηy[k]))
        Cxᵏ = abs(J.J[k])*(sfi_pu.φφξ*J.ξx[k] + sfi_pu.φφη*J.ηx[k])
        Cyᵏ = abs(J.J[k])*(sfi_pu.φφξ*J.ξy[k] + sfi_pu.φφη*J.ηy[k])

        # add to global system
        for i=1:g₂.nn
            for j=1:g₂.nn
                # eqtn 1: (∇u)⊙(∇v) term
                push!(A, (umap[1, g₂.t[k, i]], umap[1, g₂.t[k, j]], Kᵏ[i, j]))
                push!(A, (umap[2, g₂.t[k, i]], umap[2, g₂.t[k, j]], Kᵏ[i, j]))
            end
            for j=1:g₁.nn
                # eqtn 1: -p*(∇⋅v) term
                push!(A, (umap[1, g₂.t[k, i]], pmap[g₁.t[k, j]], -(Cxᵏ[i, j] + Cyᵏ[i, j])))
                push!(A, (umap[2, g₂.t[k, i]], pmap[g₁.t[k, j]], -(Cxᵏ[i, j] + Cyᵏ[i, j])))
                # eqtn 2: q*(∇⋅u)
                push!(A, (pmap[g₁.t[k, j]], umap[1, g₂.t[k, i]], Cxᵏ[i, j]))
                push!(A, (pmap[g₁.t[k, j]], umap[2, g₂.t[k, i]], Cyᵏ[i, j]))
            end
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet for u along edges
    A[umap[:, g₂.e], :] .= 0
    A[diagind(A)[umap[:, g₂.e]]] .= 1
    b[umap[1, Γ₁]] .= 0
    b[umap[2, Γ₁]] .= 0
    b[umap[1, Γ₂]] .= 1
    b[umap[2, Γ₂]] .= 0

    # set p to zero somewhere
    A[pmap[1], :] .= 0
    A[pmap[1], pmap[1]] = 1
    b[pmap[1]] = 0

    # # set p̄ to zero
    # A[pmap[1], :] .= 0
    # A[pmap[1], pmap[1:end]] .= 1
    # b[pmap[1]] = 0

    # fig, ax = subplots()
    # im = ax.imshow(abs.(Matrix(A)) .== 0, cmap="binary_r")
    # savefig("images/debug.png")
    # println("images/debug.png")
    # plt.close()
    # error()

    # println(rank(A))
    # println(N)

    # solve
    sol = A\b

    # reshape to get u and p
    return sol[umap], sol[pmap]
end

"""
    u, p = stokes_res(nref, order)
"""
function stokes_res(nref; plot=false)
    # geometry type
    geo = "square"

    # get shape functions
    sf_u = ShapeFunctions(2)
    # sf_p = ShapeFunctions(1; zeromean=true)
    sf_p = ShapeFunctions(1)

    # get shape function integrals
    sfi_uu = ShapeFunctionIntegrals(sf_u, sf_u)
    sfi_pu = ShapeFunctionIntegrals(sf_p, sf_u)

    # get grids
    g₁ = Grid("../meshes/$geo/mesh$nref.h5", 1)
    g₂ = Grid("../meshes/$geo/mesh$nref.h5", 2)

    # Γ₁: where u = 0 
    Γ₁ = g₂.e[abs.(g₂.p[g₂.e, 2] .- 1) .> 1e-4]

    # Γ₂ where u₁ = 1 
    Γ₂ = g₂.e[abs.(g₂.p[g₂.e, 2] .- 1) .< 1e-4]

    # fig, ax, im = tplot(g₂.p, g₂.t)
    # ax.plot(g₂.p[Γ₁, 1], g₂.p[Γ₁, 2], "o", ms=1)
    # ax.plot(g₂.p[Γ₂, 1], g₂.p[Γ₂, 2], "o", ms=1)
    # savefig("images/debug.png")
    # println("images/debug.png")
    # plt.close()

    # get Jacobians
    J = Jacobians(g₁)

    # # mesh resolution 
    # h = 1/sqrt(g₂.np)

    # solve stokes problem
    u, p = solve_stokes(g₁, g₂, sfi_uu, sfi_pu, J, Γ₁, Γ₂)

    if plot
        fig, ax, im = tplot(g₂.p, g₂.t, u[1, :])
        cb = colorbar(im, ax=ax, label=L"u_1")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/u1.png")
        println("images/u1.png")
        plt.close()
        fig, ax, im = tplot(g₂.p, g₂.t, u[2, :])
        cb = colorbar(im, ax=ax, label=L"u_2")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/u2.png")
        println("images/u2.png")
        plt.close()
        fig, ax, im = tplot(g₁.p, g₁.t, p)
        cb = colorbar(im, ax=ax, label=L"p")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/p.png")
        println("images/p.png")
        plt.close()
    end

    return u, p
end

u, p = stokes_res(3; plot=true)