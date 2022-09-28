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
    -Δu + ∇p = f      on Ω,
         ∇⋅u = 0      on Ω,
           u = u₀     on ∂Ω,
with extra condition
    ∫ p dx = 0.
Here u = (u₁, u₂) is the velocity vector and p is the pressure.
Weak form:
    ∫ (∇u)⊙(∇v) - p (∇⋅v) + q (∇⋅u) dx = ∫ f⋅v dx,
for all 
    v₁, v₂ ∈ P₂ and q ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes(g₁, g₂, sfi_uu, sfi_pu, J, f, u₀) 
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

        # contribution from f⋅v
        b₁ᵏ = abs(J.J[k])*sfi_uu.φφ*f[1, g₂.t[k, :]]
        b₂ᵏ = abs(J.J[k])*sfi_uu.φφ*f[2, g₂.t[k, :]]

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
            b[umap[1, g₂.t[k, i]]] += b₁ᵏ[i]
            b[umap[2, g₂.t[k, i]]] += b₂ᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet for u along edges
    A[umap[:, g₂.e], :] .= 0
    A[diagind(A)[umap[:, g₂.e]]] .= 1
    b[umap[:, g₂.e]] .= u₀

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
    h, err = stokes_res(nref)
"""
function stokes_res(nref; plot=false)
    # geometry type
    geo = "circle"

    # get shape functions
    sf_u = ShapeFunctions(2)
    sf_p = ShapeFunctions(1)

    # get shape function integrals
    sfi_uu = ShapeFunctionIntegrals(sf_u, sf_u)
    sfi_pu = ShapeFunctionIntegrals(sf_p, sf_u)
    sfi_pp = ShapeFunctionIntegrals(sf_p, sf_p)

    # get grids
    g₁ = Grid("../meshes/$geo/mesh$nref.h5", 1)
    g₂ = Grid("../meshes/$geo/mesh$nref.h5", 2)

    # mesh resolution 
    h = 1/sqrt(g₂.np)

    # exact solution
    x = g₂.p[:, 1] 
    y = g₂.p[:, 2] 
    ua₁ = @.  π/2*cos(π*x/2)*sin(π*y/2)
    ua₂ = @. -π/2*sin(π*x/2)*cos(π*y/2)
    ua = hcat(ua₁, ua₂)'
    pa = zeros(g₂.np)
    f₁ = @. π^3/4*cos(π*x/2)*sin(π*y/2)
    f₂ = @. -π^3/4*sin(π*x/2)*cos(π*y/2)
    f = hcat(f₁, f₂)'

    # dirichlet
    u₀ = hcat(ua[1, g₂.e], ua[2, g₂.e])'

    # get Jacobians
    J = Jacobians(g₁)

    # solve stokes problem
    u, p = solve_stokes(g₁, g₂, sfi_uu, sfi_pu, J, f, u₀)

    if plot
        quickplot(g₂, u[1, :], L"u_1", "images/u1.png")
        quickplot(g₂, u[2, :], L"u_2", "images/u2.png")
        quickplot(g₁, p, L"p", "images/p.png")

        quickplot(g₂, ua[1, :], L"u_1^a", "images/u1a.png")
        quickplot(g₂, ua[2, :], L"u_2^a", "images/u2a.png")
        quickplot(g₂, pa, L"p^a", "images/pa.png")

        quickplot(g₂, abs.(u[1, :] - ua[1, :]), L"|u_1 - u_1^a|", "images/e1.png")
        quickplot(g₂, abs.(u[2, :] - ua[2, :]), L"|u_2 - u_2^a|", "images/e2.png")
        quickplot(g₁, abs.(p - pa[1:g₁.np]), L"|p - p^a|", "images/ep.png")
    end

    # error
    err_u₁ = H1norm(g₂, sfi_uu, J, u[1, :] - ua[1, :])
    err_u₂ = H1norm(g₂, sfi_uu, J, u[2, :] - ua[2, :])
    err_p = L2norm(g₁, sfi_pp, J, p - pa[1:g₁.np])
    err= err_u₁ + err_u₂ + err_p
    return h, err
end

"""
    quickplot(g, u, clabel, ofile)
"""
function quickplot(g, u, clabel, ofile)
    fig, ax, im = tplot(g.p, g.t, u)
    cb = colorbar(im, ax=ax, label=clabel)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    savefig(ofile)
    println(ofile)
    plt.close()
end

"""
    stokes_convergence(nrefs)
"""
function stokes_convergence(nrefs)
    n = size(nrefs, 1)
    h = zeros(n)
    err = zeros(n)
    for i=1:n
        println(nrefs[i])
        h[i], err[i] = stokes_res(nrefs[i])
    end

    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u^a||_{H^1} + ||p - p^a||_{L^2}$")
    ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^2], "k-", label=L"$h^2$")
    ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^3], "k--", label=L"$h^3$")
    ax.loglog(h, err, "o", label="Data")
    ax.legend()
    ax.set_xlim(0.5*h[end], 2*h[1])
    ax.set_ylim(0.5*err[end], 2*err[1])
    savefig("images/stokes.png")
    println("images/stokes.png")
    plt.close()

    return h, err
end

# stokes_res(3; plot=true)
h, err = stokes_convergence(0:5)