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
    uˣ, uᶻ, p = solve_stokes_hydro(g, s, J, b, e)

stokes_hydro problem:
    -∂zz(uˣ) + ∂x(p) = 0,
               ∂z(p) = b,
     ∂x(uˣ) + ∂z(uᶻ) = 0, 
with extra condition
    ∫ p dx dz = 0.
Boundary conditions are 
       uˣ = uᶻ = 0 at z = -H,
        ∂z(uˣ) = 0 at z = 0, 
            uᶻ = 0 at z = 0,
Weak form:
    ∫ [ ∂z(uˣ)∂z(vˣ) - p∂x(vˣ) 
      - p∂z(vᶻ)
      + q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ bvᶻ dx dz,
for all 
    uˣ, vˣ ∈ Pᵢ,
    uᶻ, vᶻ ∈ Pⱼ, 
    p, q ∈ Pₖ
where j = i-1, k = i-2, and Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_hydro(g, s, J, b, e)
    # indices
    uˣmap = 1:g.u.np
    uᶻmap = uˣmap[end] .+ (1:g.w.np)
    pmap  = uᶻmap[end] .+ (1:g.p.np)
    N = pmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g.p.nt
        # ∂z(uˣ)∂z(vˣ)
        Kᵏ = abs(J.J[k])*(s.uu.φξφξ*J.ξy[k]^2 + s.uu.φξφη*J.ξy[k]*J.ηy[k] + s.uu.φηφξ*J.ηy[k]*J.ξy[k] + s.uu.φηφη*J.ηy[k]^2)

        # p*∂x(vˣ) 
        Cx_puᵏ = abs(J.J[k])*(s.pu.φφξ*J.ξx[k] + s.pu.φφη*J.ηx[k])
        # p*∂z(vᶻ)
        Cz_pwᵏ = abs(J.J[k])*(s.pw.φφξ*J.ξy[k] + s.pw.φφη*J.ηy[k])
        # q*∂x(uˣ) 
        Cx_upᵏ = abs(J.J[k])*(s.up.φξφ*J.ξx[k] + s.up.φηφ*J.ηx[k])
        # q*∂z(uᶻ)
        Cz_wpᵏ = abs(J.J[k])*(s.wp.φξφ*J.ξy[k] + s.wp.φηφ*J.ηy[k])

        # δ*q*p
        Mppᵏ = abs(J.J[k])*s.pp.φφ

        # b*vᶻ
        rᵏ = abs(J.J[k])*s.ww.φφ*b[g.w.t[k, :]]

        # uˣ*vˣ
        for i=1:g.u.nn, j=1:g.u.nn
            # x-mom: ∂z(uˣ)∂z(vˣ)
            push!(A, (uˣmap[g.u.t[k, i]], uˣmap[g.u.t[k, j]], Kᵏ[i, j]))
        end
        # p*vˣ
        for i=1:g.u.nn, j=1:g.p.nn
            # x-mom: -p*∂x(vˣ)
            push!(A, (uˣmap[g.u.t[k, i]], pmap[g.p.t[k, j]], -Cx_puᵏ[i, j]))
        end
        # uˣ*q
        for i=1:g.p.nn, j=1:g.u.nn
            # cont: ∂x(uˣ)*q
            push!(A, (pmap[g.p.t[k, i]], uˣmap[g.u.t[k, j]], Cx_upᵏ[i, j]))
        end
        # p*vᶻ
        for i=1:g.w.nn, j=1:g.p.nn
            # z-mom: -p*∂z(vᶻ)
            push!(A, (uᶻmap[g.w.t[k, i]], pmap[g.p.t[k, j]], -Cz_pwᵏ[i, j]))
        end
        # uᶻ*q
        for i=1:g.p.nn, j=1:g.w.nn
            # cont: ∂z(uᶻ)*q
            push!(A, (pmap[g.p.t[k, i]], uᶻmap[g.w.t[k, j]], Cz_wpᵏ[i, j]))
        end
        # p*p
        for i=1:g.p.nn, j=1:g.p.nn
            # pressure condition: δ*q*p
            push!(A, (pmap[g.p.t[k, i]], pmap[g.p.t[k, j]], 1e-7*Mppᵏ[i, j]))
        end
        # b
        r[uᶻmap[g.w.t[k, :]]] .+= rᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # uˣ = uᶻ = 0 at z = -H (replace mom eqtns at bottom bdy)
    A[uˣmap[e.botu], :] .= 0
    A[diagind(A)[uˣmap[e.botu]]] .= 1
    r[uˣmap[e.botu]] .= 0

    A[uᶻmap[e.botw], :] .= 0
    A[diagind(A)[uᶻmap[e.botw]]] .= 1
    r[uᶻmap[e.botw]] .= 0

    # ∂z(uˣ) = 0 at z = 0 → natural
    r[uˣmap[e.topu]] .= 1

    # uᶻ = 0 at z = 0 (replace mom eqtn at top bdy)
    A[uᶻmap[e.topw], :] .= 0
    A[diagind(A)[uᶻmap[e.topw]]] .= 1
    r[uᶻmap[e.topw]] .= 0

    # solve
    sol = A\r

    # reshape to get u and p
    return sol[uˣmap], sol[uᶻmap], sol[pmap]
end

"""
    h, err = stokes_hydro_res(nref)
"""
function stokes_hydro_res(nref, order; plot=false)
    # geometry type
    # geo = "jc"
    geo = "gmsh"

    # get shape functions
    sp = ShapeFunctions(order-2)
    sw = ShapeFunctions(order-1)
    su = ShapeFunctions(order)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(su, su)
    ww = ShapeFunctionIntegrals(sw, sw)
    pu = ShapeFunctionIntegrals(sp, su)
    pw = ShapeFunctionIntegrals(sp, sw)
    up = ShapeFunctionIntegrals(su, sp)
    wp = ShapeFunctionIntegrals(sw, sp)
    pp = ShapeFunctionIntegrals(sp, sp)
    s = (uu = uu,
         ww = ww, 
         pu  = pu,  
         pw  = pw,  
         up  = up,  
         wp  = wp,
         pp   = pp)  

    # get grids
    gp = Grid("../meshes/$geo/mesh$nref.h5", order - 2)
    gw = Grid("../meshes/$geo/mesh$nref.h5", order - 1)
    gu = Grid("../meshes/$geo/mesh$nref.h5", order)
    g = (p = gp, u = gu, w = gw)

    # top and bottom edges
    ebotw, etopw = get_sides(gw)
    ebotu, etopu = get_sides(gu)
    e = (botw = ebotw, topw = etopw, 
         botu = ebotu, topu = etopu)

    # forcing
    x = g.w.p[:, 1] 
    z = g.w.p[:, 2] 
    b = zeros(g.w.np)
    # b = @. exp(-x^2/0.1^2 - (z + 0.2)^2/0.1^2)
    # H_func(x) = sqrt(2 - x^2) - 1
    # H_func(x) = 1 - x^2
    # H = H_func.(x)
    # δ = 0.2
    # b = @. z + δ*H*exp(-(z/H + 1)/δ)
    # b[H .== 0] .= 0
    # b = z

    # get Jacobians
    g1 = Grid("../meshes/$geo/mesh$nref.h5", 1)
    J = Jacobians(g1)

    # solve stokes_hydro problem
    uˣ, uᶻ, p = solve_stokes_hydro(g, s, J, b, e)

    if plot
        quickplot(g.w, b, g.u, uˣ, L"u^x", "images/ux.png")
        quickplot(g.w, b, g.w, uᶻ, L"u^z", "images/uz.png")
        quickplot(g.w, b, g.w, b, L"b", "images/b.png")
        quickplot(g.w, b, g.w, p, L"p", "images/p.png")
    end

    return uˣ, uᶻ, p
end

uˣ, uᶻ, p = stokes_hydro_res(3, 2; plot=true)

println("Done.")