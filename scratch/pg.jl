using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

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
    ∫ [ε²∂z(uˣ)∂z(v₁) - uʸv₁ + ∂x(p)v₁ +
       ε²∂z(uʸ)∂z(v₂) + uˣv₂ +
       ∂z(p)v₃ +
        q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ bv₃ dx dz,
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

        ###
        K11ᵏ = abs(J.J[k])*(s11.φξφξ*J.ξy[k]^2 + s11.φξφη*J.ξy[k]*J.ηy[k] + s11.φηφξ*J.ηy[k]*J.ξy[k] + s11.φηφη*J.ηy[k]^2)
        ###

        # uv terms
        Mᵏ = abs(J.J[k])*s22.φφ

        # p*∂x(v) and p*∂z(v) terms (also q*∂x(u) and q*∂z(u))
        Cxᵏ = abs(J.J[k])*(s12.φξφ*J.ξx[k] + s12.φηφ*J.ηx[k])
        Czᵏ = abs(J.J[k])*(s11.φξφ*J.ξy[k] + s11.φηφ*J.ηy[k])

        # b*v term
        rᵏ = abs(J.J[k])*s11.φφ*b[g1.t[k, :]]

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
                # ∂x(p)v₁ 
                push!(A, (uˣmap[g2.t[k, i]], pmap[g1.t[k, j]], Cxᵏ[i, j]))
                # ∂x(uˣ)q
                push!(A, (pmap[g1.t[k, j]], uˣmap[g2.t[k, i]], Cxᵏ[i, j]))
            end
        end
        # s1*s1
        for i=1:g1.nn
            for j=1:g1.nn
                # ∂z(p)v₃
                push!(A, (uᶻmap[g1.t[k, i]], pmap[g1.t[k, j]], Czᵏ[i, j]))
                # ∂z(uᶻ)q
                push!(A, (pmap[g1.t[k, j]], uᶻmap[g1.t[k, i]], Czᵏ[i, j]))

                ###
                push!(A, (uᶻmap[g2.t[k, i]], uᶻmap[g2.t[k, j]], ε²*K11ᵏ[i, j]))
                ###
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

    # uᶻ = 0 at z = 0 (replace continuity at top bdy)
    A[pmap[etop1], :] .= 0
    for e in etop1
        A[pmap[e], uᶻmap[e]] = 1
    end
    r[pmap[etop1]] .= 0

    # p constraint (replace one of the uᶻ = 0 conditions at top bdy)
    n = convert(Int64, round(size(etop1, 1)/2)) # middle of top bdy
    i = pmap[etop1[n]]
    A[i, :] .= 0
    A[i, i] = 1
    r[i] = 0

    # println(N)
    # println(rank(A))

    # solve
    sol = A\r

    # reshape to get uˣ, uʸ, uᶻ and p
    return sol[uˣmap], sol[uʸmap], sol[uᶻmap], sol[pmap]
end

"""
    h, err = pg_res(nref, order)
"""
function pg_res(nref, order; plot=false)
    # Ekman number
    ε² = 1e-3

    # geometry type
    geo = "jc"

    # get shape functions
    sf2 = ShapeFunctions(order + 1)
    sf1 = ShapeFunctions(order)

    # get shape function integrals
    s22 = ShapeFunctionIntegrals(sf2, sf2)
    s12 = ShapeFunctionIntegrals(sf1, sf2)
    s11 = ShapeFunctionIntegrals(sf1, sf1)

    # get grids
    g0 = Grid("../meshes/$geo/mesh$nref.h5", 1)
    g1 = Grid("../meshes/$geo/mesh$nref.h5", order)
    g2 = Grid("../meshes/$geo/mesh$nref.h5", order + 1)

    # mesh resolution 
    h = 1/sqrt(g2.np)

    # top and bottom edges
    ebot1, etop1 = get_sides(g1)
    ebot2, etop2 = get_sides(g2)

    # buoyancy field
    x = g1.p[:, 1] 
    z = g1.p[:, 2] 
    # b = @. exp(-x^2/0.1^2 - (z + 0.2)^2/0.1^2)
    # b = @. exp(-x^2/0.1^2 - (z + 0.4)^2/0.1^2)
    H_func(x) = lerp(g1.p[ebot1, 1], -g1.p[ebot1, 2], x)
    H = H_func.(x)
    δ = 0.2
    b = @. z + δ*H*exp(-(z/H + 1)/δ)
    b[H .== 0] .= 0

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
        x = g1.p[ebot1, 1]
        H = H_func.(x)
        quickplot(x, H, g1, b, g2, uˣ, L"u^x", "images/ux.png")
        quickplot(x, H, g1, b, g2, uʸ, L"u^y", "images/uy.png")
        quickplot(x, H, g1, b, g1, uᶻ, L"u^z", "images/uz.png")
        quickplot(x, H, g1, b, g1, p, L"p", "images/p.png")
        quickplot(x, H, g1, b, g1, b, L"b", "images/b.png")
    end

    # error
    err = 0
    return h, err
end

pg_res(3, 1; plot=true)