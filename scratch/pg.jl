using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    uˣ, uʸ, uᶻ, p = solve_pg(g₁, g₂, sfi_uu, sfi_pu, J, b, ε², ebot, etop)

PG Inversion:
    -ε²∂zz(uˣ) - uʸ + ∂x(p) = 0 
    -ε²∂zz(uʸ) + uˣ         = 0
                      ∂z(p) = b
            ∂x(uˣ) + ∂z(uᶻ) = 0
with extra condition
    ∫ p dx dy = 0.
Boundary conditions are 
       uˣ = uʸ = uᶻ = 0 at z = -H
    ∂z(uˣ) = ∂z(uʸ) = 0 at z = 0 
                 uᶻ = 0 at z = 0
Weak form:
    ∫ [ε²∂z(uˣ)∂z(v₁) - uʸv₁ - p∂x(v₁) +
       ε²∂z(uʸ)∂z(v₂) + uˣv₂ +
       -p∂z(v₃) +
        q∂x(uˣ) + q∂z(uᶻ)
      ] dx dy
    = ∫ b v₃ dx dy
for all 
    v₁, v₂, v₃ ∈ P₂ and q ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_pg(g₁, g₂, sfi_uu, sfi_pu, J, b, ε², ebot, etop) 
    # indices
    umap = reshape(1:3*g₂.np, (3, g₂.np))
    pmap = umap[end] .+ (1:g₁.np)
    N = pmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g₁.nt
        # ∂z(u)∂z(v) terms
        Kᵏ = abs(J.J[k])*(sfi_uu.φξφξ*J.ξy[k]^2 + 
                          sfi_uu.φξφη*J.ξy[k]*J.ηy[k] +
                          sfi_uu.φηφξ*J.ηy[k]*J.ξy[k] +
                          sfi_uu.φηφη*J.ηy[k]^2)

        # uv terms
        Mᵏ = abs(J.J[k])*sfi_uu.φφ

        # p*∂x(v) and p*∂z(v) terms (also q*∂x(u) and q*∂z(u))
        Cxᵏ = abs(J.J[k])*(sfi_pu.φφξ*J.ξx[k] + sfi_pu.φφη*J.ηx[k])
        Czᵏ = abs(J.J[k])*(sfi_pu.φφξ*J.ξy[k] + sfi_pu.φφη*J.ηy[k])

        # b*v term
        rᵏ = abs(J.J[k])*sfi_uu.φφ*b[g₂.t[k, :]]

        # add to global system
        for i=1:g₂.nn
            for j=1:g₂.nn
                # ∂z(u)∂z(v) terms
                push!(A, (umap[1, g₂.t[k, i]], umap[1, g₂.t[k, j]], ε²*Kᵏ[i, j]))
                push!(A, (umap[2, g₂.t[k, i]], umap[2, g₂.t[k, j]], ε²*Kᵏ[i, j]))
                push!(A, (umap[3, g₂.t[k, i]], umap[3, g₂.t[k, j]], ε²*Kᵏ[i, j]))
                # uv terms
                push!(A, (umap[1, g₂.t[k, i]], umap[2, g₂.t[k, j]], -Mᵏ[i, j]))
                push!(A, (umap[2, g₂.t[k, i]], umap[1, g₂.t[k, j]], Mᵏ[i, j]))
            end
            for j=1:g₁.nn
                # p*∂x(v) and p*∂z(v) terms
                push!(A, (umap[1, g₂.t[k, i]], pmap[g₁.t[k, j]], -Cxᵏ[i, j]))
                push!(A, (umap[3, g₂.t[k, i]], pmap[g₁.t[k, j]], -Czᵏ[i, j]))
                # q*∂x(u) and q*∂z(u) terms
                push!(A, (pmap[g₁.t[k, j]], umap[1, g₂.t[k, i]], Cxᵏ[i, j]))
                push!(A, (pmap[g₁.t[k, j]], umap[3, g₂.t[k, i]], Czᵏ[i, j]))
            end
            r[umap[3, g₂.t[k, i]]] += rᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # uˣ = uʸ = uᶻ = 0 at z = -H
    A[umap[:, ebot], :] .= 0
    A[diagind(A)[umap[:, ebot]]] .= 1
    r[umap[:, ebot]] .= 0

    # ∂z(uˣ) = ∂(uʸ) = 0 at z = 0 → natural

    # uᶻ = 0 at z = 0
    A[umap[3, etop], :] .= 0
    A[diagind(A)[umap[3, etop]]] .= 1
    r[umap[3, etop]] .= 0

    # set p to zero somewhere
    A[pmap[1], :] .= 0
    A[pmap[1], pmap[1]] = 1
    r[pmap[1]] = 0

    # println(N)
    # println(rank(A))

    # solve
    sol = A\r

    # reshape to get uˣ, uʸ, uᶻ and p
    # return sol[umap[1, :]], sol[umap[2, :]], sol[umap[3, :]], sol[pmap]
    return sol[umap[1, :]], sol[umap[2, :]], sol[umap[2, :]], sol[pmap]
end

"""
    h, err = pg_res(nref)
"""
function pg_res(nref; plot=false)
    # Ekman number
    ε² = 1

    # geometry type
    geo = "jc"

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

    # buoyancy field
    x = g₂.p[:, 1] 
    z = g₂.p[:, 2] 
    b = @. -exp(-x^2/0.1 - (z + 0.2)^2/0.1)

    # top and bottom edges
    etop = g₂.e[abs.(g₂.p[g₂.e, 2]) .< 1e-4]
    ebot = g₂.e[abs.(g₂.p[g₂.e, 2]) .>= 1e-4]
    eleft = g₂.e[abs.(g₂.p[g₂.e, 1] .+ 1) .<= 1e-4]
    eright = g₂.e[abs.(g₂.p[g₂.e, 1] .- 1) .<= 1e-4]
    deleteat!(etop, findall(x->x==eleft[1],etop))
    deleteat!(etop, findall(x->x==eright[1],etop))
    push!(ebot, eleft[1])
    push!(ebot, eright[1])

    # fig, ax, im = tplot(g₂.p, g₂.t)
    # ax.plot(g₂.p[ebot, 1], g₂.p[ebot,2], "o", ms=1)
    # ax.plot(g₂.p[etop, 1], g₂.p[etop,2], "o", ms=1)
    # # ax.set_xlim(-1.1, -0.9)
    # ax.set_xlim(0.9, 1.1)
    # ax.set_ylim(-0.1, 0.0)
    # savefig("images/debug.png")
    # plt.close()
    # error()

    # # exact solution
    # x = g₂.p[:, 1] 
    # y = g₂.p[:, 2] 
    # ua₁ = @.  π/2*cos(π*x/2)*sin(π*y/2)
    # ua₂ = @. -π/2*sin(π*x/2)*cos(π*y/2)
    # ua = hcat(ua₁, ua₂)'
    # pa = zeros(g₂.np)
    # f₁ = @. π^3/4*cos(π*x/2)*sin(π*y/2)
    # f₂ = @. -π^3/4*sin(π*x/2)*cos(π*y/2)
    # f = hcat(f₁, f₂)'

    # get Jacobians
    J = Jacobians(g₁)

    # solve stokes problem
    uˣ, uʸ, uᶻ, p = solve_pg(g₁, g₂, sfi_uu, sfi_pu, J, b, ε², ebot, etop)

    if plot
        quickplot(g₂, uˣ, L"u^x", "images/ux.png")
        quickplot(g₂, uʸ, L"u^y", "images/uy.png")
        quickplot(g₂, uᶻ, L"u^z", "images/uz.png")
        quickplot(g₁, p, L"p", "images/p.png")
        quickplot(g₂, b, L"b", "images/b.png")
    end

    # error
    err = 0
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
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end

pg_res(3; plot=true)