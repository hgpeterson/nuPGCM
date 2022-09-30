using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    uˣ, uʸ, uᶻ, p = solve_pg(g1, g2, sfi_uu, sfi_pu, J, b, ε², ebot, etop)

PG Inversion:
    -ε²∂zz(uˣ) - uʸ + ∂x(p) = 0, 
    -ε²∂zz(uʸ) + uˣ         = 0,
                      ∂z(p) = b,
            ∂x(uˣ) + ∂z(uᶻ) = 0,
with extra condition
    ∫ p dx dy = 0.
Boundary conditions are 
       uˣ = uʸ = uᶻ = 0 at z = -H,
    ∂z(uˣ) = ∂z(uʸ) = 0 at z = 0, 
                 uᶻ = 0 at z = 0,
Weak form:
    ∫ [ε²∂z(uˣ)∂z(v₁) - uʸv₁ - p∂x(v₁) +
       ε²∂z(uʸ)∂z(v₂) + uˣv₂ +
       -p∂z(v₃) +
        q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ b v₃ dx dz,
for all 
    v₁, v₂ ∈ P₂ and q, v₃ ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_pg(g1, g2, s22, s12, s11, J, b, ε², ebot1, ebot2, etop1) 
    # indices
    uˣmap = 1:g2.np
    uʸmap = uˣmap[end] .+ (1:g2.np)
    uᶻmap = uʸmap[end] .+ (1:g1.np)
    pmap  = uᶻmap[end] .+ (1:g1.np)
    N = pmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # ∂z(u)∂z(v) terms
        Kᵏ = abs(J.J[k])*(s22.φξφξ*J.ξy[k]^2 + s22.φξφη*J.ξy[k]*J.ηy[k] + s22.φηφξ*J.ηy[k]*J.ξy[k] + s22.φηφη*J.ηy[k]^2)
        # K11ᵏ = abs(J.J[k])*(s11.φξφξ*J.ξy[k]^2 + s11.φξφη*J.ξy[k]*J.ηy[k] + s11.φηφξ*J.ηy[k]*J.ξy[k] + s11.φηφη*J.ηy[k]^2)

        # uv terms
        Mᵏ = abs(J.J[k])*s22.φφ

        # p*∂x(v) and p*∂z(v) terms (also q*∂x(u) and q*∂z(u))
        Cxᵏ = abs(J.J[k])*(s12.φφξ*J.ξx[k] + s12.φφη*J.ηx[k])
        Czᵏ = abs(J.J[k])*(s11.φφξ*J.ξy[k] + s11.φφη*J.ηy[k])

        # b*v term
        rᵏ = abs(J.J[k])*s12.φφ'*b[g2.t[k, :]]

        # s2*s2
        for i=1:g2.nn
            for j=1:g2.nn
                # ∂z(u)∂z(v) terms
                push!(A, (uˣmap[g2.t[k, i]], uˣmap[g2.t[k, j]], ε²*Kᵏ[i, j]))
                push!(A, (uʸmap[g2.t[k, i]], uʸmap[g2.t[k, j]], ε²*Kᵏ[i, j]))
                # uv terms
                push!(A, (uˣmap[g2.t[k, i]], uʸmap[g2.t[k, j]], -Mᵏ[i, j]))
                push!(A, (uʸmap[g2.t[k, j]], uˣmap[g2.t[k, i]], Mᵏ[i, j]))
            end
        end
        # s1*s2
        for i=1:g2.nn
            for j=1:g1.nn
                # p*∂x(v) 
                push!(A, (uˣmap[g2.t[k, i]], pmap[g1.t[k, j]], -Cxᵏ[i, j]))
                # q*∂x(u) 
                push!(A, (pmap[g1.t[k, j]], uˣmap[g2.t[k, i]], Cxᵏ[i, j]))
            end
        end
        # s1*s1
        for i=1:g1.nn
            for j=1:g1.nn
                # # ∂z(u)∂z(v) terms
                # push!(A, (uᶻmap[g1.t[k, i]], uᶻmap[g1.t[k, j]], ε²*K11ᵏ[i, j]))
                # p*dz(u)
                push!(A, (uᶻmap[g1.t[k, i]], pmap[g1.t[k, j]], -Czᵏ[i, j]))
                # q*dz(v)
                push!(A, (pmap[g1.t[k, j]], uᶻmap[g1.t[k, i]], Czᵏ[i, j]))
            end
            r[uᶻmap[g1.t[k, i]]] += rᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # uˣ = uʸ = uᶻ = 0 at z = -H
    A[uˣmap[ebot2], :] .= 0
    A[diagind(A)[uˣmap[ebot2]]] .= 1
    r[uˣmap[ebot2]] .= 0

    A[uʸmap[ebot2], :] .= 0
    A[diagind(A)[uʸmap[ebot2]]] .= 1
    r[uʸmap[ebot2]] .= 0

    A[uᶻmap[ebot1], :] .= 0
    A[diagind(A)[uᶻmap[ebot1]]] .= 1
    r[uᶻmap[ebot1]] .= 0

    # ∂z(uˣ) = ∂(uʸ) = 0 at z = 0 → natural

    # uᶻ = 0 at z = 0
    A[uᶻmap[etop1], :] .= 0
    A[diagind(A)[uᶻmap[etop1]]] .= 1
    r[uᶻmap[etop1]] .= 0

    # set p to zero somewhere
    A[pmap[1], :] .= 0
    A[pmap[1], pmap[1]] = 1
    r[pmap[1]] = 0

    # println(N)
    # println(rank(A))

    # solve
    sol = A\r

    # reshape to get uˣ, uʸ, uᶻ and p
    return sol[uˣmap], sol[uʸmap], sol[uᶻmap], sol[pmap]
end

"""
    h, err = pg_res(nref)
"""
function pg_res(nref; plot=false)
    # Ekman number
    ε² = 0.0001

    # geometry type
    geo = "jc"

    # get shape functions
    sf2 = ShapeFunctions(2)
    sf1 = ShapeFunctions(1)

    # get shape function integrals
    s22 = ShapeFunctionIntegrals(sf2, sf2)
    s12 = ShapeFunctionIntegrals(sf1, sf2)
    s11 = ShapeFunctionIntegrals(sf1, sf1)

    # get grids
    g1 = Grid("../meshes/$geo/mesh$nref.h5", 1)
    g2 = Grid("../meshes/$geo/mesh$nref.h5", 2)

    # mesh resolution 
    h = 1/sqrt(g2.np)

    # top and bottom edges
    etop1 = g1.e[abs.(g1.p[g1.e, 2]) .< 1e-4]
    ebot1 = g1.e[abs.(g1.p[g1.e, 2]) .>= 1e-4]
    eleft1 = g1.e[abs.(g1.p[g1.e, 1] .+ 1) .<= 1e-4]
    eright1 = g1.e[abs.(g1.p[g1.e, 1] .- 1) .<= 1e-4]
    deleteat!(etop1, findall(x->x==eleft1[1], etop1))
    deleteat!(etop1, findall(x->x==eright1[1], etop1))
    push!(ebot1, eleft1[1])
    push!(ebot1, eright1[1])

    ebot2 = g2.e[abs.(g2.p[g2.e, 2]) .>= 1e-4]
    eleft2 = g2.e[abs.(g2.p[g2.e, 1] .+ 1) .<= 1e-4]
    eright2 = g2.e[abs.(g2.p[g2.e, 1] .- 1) .<= 1e-4]
    push!(ebot2, eleft2[1])
    push!(ebot2, eright2[1])

    # buoyancy field
    x = g2.p[:, 1] 
    z = g2.p[:, 2] 
    # b = @. exp(-x^2/0.1^2 - (z + 0.2)^2/0.1^2)
    # b = @. exp(-x^2/0.1^2 - (z + 0.4)^2/0.1^2)
    H = @. sqrt(2 - x^2) - 1
    b = @. z + 0.1*H*exp(-(z + H)/(0.1*H))
    b[ebot2] .= H[ebot2]

    # fig, ax, im = tplot(g2.p, g2.t)
    # ax.plot(g2.p[ebot, 1], g2.p[ebot,2], "o", ms=1)
    # ax.plot(g2.p[etop, 1], g2.p[etop,2], "o", ms=1)
    # # ax.set_xlim(-1.1, -0.9)
    # ax.set_xlim(0.9, 1.1)
    # ax.set_ylim(-0.1, 0.0)
    # savefig("images/debug.png")
    # plt.close()
    # error()

    # # exact solution
    # x = g2.p[:, 1] 
    # y = g2.p[:, 2] 
    # ua₁ = @.  π/2*cos(π*x/2)*sin(π*y/2)
    # ua₂ = @. -π/2*sin(π*x/2)*cos(π*y/2)
    # ua = hcat(ua₁, ua₂)'
    # pa = zeros(g2.np)
    # f₁ = @. π^3/4*cos(π*x/2)*sin(π*y/2)
    # f₂ = @. -π^3/4*sin(π*x/2)*cos(π*y/2)
    # f = hcat(f₁, f₂)'

    # get Jacobians
    J = Jacobians(g1)

    # solve stokes problem
    uˣ, uʸ, uᶻ, p = solve_pg(g1, g2, s22, s12, s11, J, b, ε², ebot1, ebot2, etop1)

    if plot
        quickplot(g2, b, g2, uˣ, L"u^x", "images/ux.png")
        quickplot(g2, b, g2, uʸ, L"u^y", "images/uy.png")
        quickplot(g2, b, g1, uᶻ, L"u^z", "images/uz.png")
        quickplot(g2, b, g1, p, L"p", "images/p.png")
        quickplot(g2, b, g2, b, L"b", "images/b.png")
    end

    # error
    err = 0
    return h, err
end

"""
    quickplot(g, u, clabel, ofile)
"""
function quickplot(gb, b, gu, u, clabel, ofile)
    fig, ax, im = tplot(gu.p, gu.t, u)
    cb = colorbar(im, ax=ax, label=clabel, orientation="horizontal", pad=0.25)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    # levels = -0.35:0.05:-0.05
    ax.tricontour(gb.p[:, 1], gb.p[:, 2], gb.t[:, 1:3] .- 1, b, #levels=levels,
                  linewidths=0.5, colors="k", linestyles="-", alpha=0.3)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    savefig(ofile)
    println(ofile)
    plt.close()
end

pg_res(3; plot=true)