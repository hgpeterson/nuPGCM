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
    ∫ (∇u)⊙(∇v) - p (∇⋅v) + q (∇⋅u) dx = 0,
for all 
    v₁, v₂ ∈ P₂ and q ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes(g₁, g₂, sfi_uu, sfi_pu, J, Γ₁, Γ₂) 
    # indices
    umap = reshape(1:2*g₂.np, (2, g₂.np))
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
        Cxᵏ = abs(J.J[k])*(sfi_pu.φφξ*J.ξx[k] + sfi_pu.φφη*J.ηx[k])
        Cyᵏ = abs(J.J[k])*(sfi_pu.φφξ*J.ξy[k] + sfi_pu.φφη*J.ηy[k])

        # add to global system
        for i=1:g₂.nn
            for j=1:g₂.nn
                # (∇u)⊙(∇v) term
                push!(A, (umap[1, g₂.t[k, i]], umap[1, g₂.t[k, j]], Kᵏ[i, j]))
                push!(A, (umap[2, g₂.t[k, i]], umap[2, g₂.t[k, j]], Kᵏ[i, j]))
            end
            for j=1:g₁.nn
                # -p*(∇⋅v) term
                push!(A, (umap[1, g₂.t[k, i]], pmap[g₁.t[k, j]], -Cxᵏ[i, j]))
                push!(A, (umap[2, g₂.t[k, i]], pmap[g₁.t[k, j]], -Cyᵏ[i, j]))
                # q*(∇⋅u) term (i and j flipped because we used sfi_pu)
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

    # fig, ax = subplots(1)
    # im = ax.imshow(abs.(Matrix(A)) .== 0, cmap="binary_r")
    # savefig("images/A.png")
    # println("images/A.png")
    # plt.close()

    # solve
    sol = A\b

    # reshape to get u and p
    return sol[umap], sol[pmap]
end

"""
    g₁, g₂, u, p = stokes_res(nref)
"""
function stokes_res(nref; plot=false)
    # geometry type
    geo = "square"

    # get shape functions
    sf_u = ShapeFunctions(2)
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

    return g₁, g₂, u, p
end

"""
    stokes_convergence(nrefs)
"""
function stokes_convergence(nrefs)
    # compare to fine res
    g₁_fine, g₂_fine, u_fine, p_fine = stokes_res(nrefs[end])

    # save h's and errors
    n = size(nrefs, 1)
    hs_u = zeros(n-1)
    hs_p = zeros(n-1)
    err_u = zeros(n-1)
    err_p = zeros(n-1)
    sf_u = ShapeFunctions(2)
    sf_p = ShapeFunctions(1)
    for i=1:n-1
        println(nrefs[i])

        # solve
        g₁, g₂, u, p = stokes_res(nrefs[i])

        # resolution 
        hs_u[i] = 1/sqrt(g₂.np)
        hs_p[i] = 1/sqrt(g₁.np)

        # error
        for j=1:g₁_fine.np
            if j in g₁_fine.e
                continue
            end
            err_p[i] += (p_fine[j] - fem_evaluate(p, g₁_fine.p[j, :], g₁, sf_p))^2
        end
        for j=1:g₂_fine.np
            if j in g₂_fine.e
                continue
            end
            err_u[i] += (u_fine[1, j] - fem_evaluate(u[1, :], g₂_fine.p[j, :], g₂, sf_u))^2
            err_u[i] += (u_fine[2, j] - fem_evaluate(u[2, :], g₂_fine.p[j, :], g₂, sf_u))^2
        end
    end
    err_p = sqrt.(err_p)
    err_u = sqrt.(err_u)

    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u_0||_{L^2}$")
    # ax.set_ylabel(L"Error $||u - u_a||_{L^\infty}$")
    # ax.loglog([hs_l[1], hs_l[end]], [err_l[1], err_l[1]*(hs_l[end]/hs_l[1])^2], "k-",  label=L"$h^2$")
    # ax.loglog([hs_q[1], hs_q[end]], [err_q[1], err_q[1]*(hs_q[end]/hs_q[1])^3], "k--", label=L"$h^3$")
    ax.loglog(hs_u, err_u, "o", label=L"u")
    ax.loglog(hs_p, err_p, "o", label=L"p")
    ax.legend(ncol=2)
    # ax.set_xlim(0.9*hs_q[end], 1.1*hs_l[1])
    # ax.set_ylim(0.5*err_q[end], 2*err_l[1])
    savefig("images/stokes.png")
    println("images/stokes.png")
    plt.close()
end

stokes_res(3; plot=true)
# stokes_convergence(0:4)