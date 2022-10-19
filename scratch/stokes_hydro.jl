using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")

Line2D = pyimport("matplotlib.lines").Line2D
plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    uˣ, uᶻ, p = solve_stokes_hydro(g, s, J, b, e, f, u₀)

stokes_hydro problem:
    -∂zz(uˣ) + ∂x(p) = fˣ,
               ∂z(p) = fᶻ,
     ∂x(uˣ) + ∂z(uᶻ) = 0, 
with extra condition
    ∫ p dx dz = 0,
and Dirichlet boundary conditions on u.
Weak form:
    ∫ [ ∂z(uˣ)∂z(vˣ) - p∂x(vˣ) 
      - p∂z(vᶻ)
      + q∂x(uˣ) + q∂z(uᶻ)
      ] dx dz
    = ∫ [fˣvˣ + fᶻvᶻ] dx dz,
for all 
    uˣ, vˣ ∈ Pᵢ,
    uᶻ, vᶻ ∈ Pⱼ, 
    p, q ∈ Pₖ
where j = i-1, k = i-2, and Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_hydro(g, s, J, e, f, u₀, p₀; diri_mask=(true, true, true, true))
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

        # # δ*q*p
        # Mppᵏ = abs(J.J[k])*s.pp.φφ

        # fˣ*vˣ
        rxᵏ = abs(J.J[k])*s.uu.φφ*f.x[g.u.t[k, :]]
        # fᶻ*vᶻ
        rzᵏ = abs(J.J[k])*s.ww.φφ*f.z[g.w.t[k, :]]

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
        # # p*p
        # for i=1:g.p.nn, j=1:g.p.nn
        #     # pressure condition: δ*q*p
        #     push!(A, (pmap[g.p.t[k, i]], pmap[g.p.t[k, j]], 1e-7*Mppᵏ[i, j]))
        # end
        # f
        r[uˣmap[g.u.t[k, :]]] .+= rxᵏ
        r[uᶻmap[g.w.t[k, :]]] .+= rzᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet condition on bottom and top (replace mom eqtns)
    if diri_mask[1]
        A[uˣmap[e.botu], :] .= 0
        A[diagind(A)[uˣmap[e.botu]]] .= 1
        r[uˣmap[e.botu]] .= u₀.botu
    end
    if diri_mask[2]
        A[uᶻmap[e.botw], :] .= 0
        A[diagind(A)[uᶻmap[e.botw]]] .= 1
        r[uᶻmap[e.botw]] .= u₀.botw
    end
    if diri_mask[3]
        A[uˣmap[e.topu], :] .= 0
        A[diagind(A)[uˣmap[e.topu]]] .= 1
        r[uˣmap[e.topu]] .= u₀.topu
    end
    if diri_mask[4]
        A[uᶻmap[e.topw], :] .= 0
        A[diagind(A)[uᶻmap[e.topw]]] .= 1
        r[uᶻmap[e.topw]] .= u₀.topw
    end

    # if N < 1000
    #     fig, ax = subplots(1)
    #     ax.imshow(abs.(Matrix(A)) .== 0, cmap="binary_r")
    #     ax.spines["left"].set_visible(false)
    #     ax.spines["bottom"].set_visible(false)
    #     savefig("images/A.png")
    #     println("images/A.png")
    #     plt.close()
    #     # println("Condition number: ", cond(Array(A)))
    # end

    # pressure condition
    A[pmap[1], :] .= 0
    A[pmap[1], pmap[1]] = 1
    r[pmap[1]] = p₀

    # solve
    println("N = $N")
    sol = A\r

    # reshape to get u and p
    return sol[uˣmap], sol[uᶻmap], sol[pmap]
end

"""
    hs, errs = stokes_hydro_res(nref)
"""
function stokes_hydro_res(nref; plot=false)
    # order of polynomials
    order = 2

    # geometry type
    geo = "jc"
    # geo = "gmsh"

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
         pu = pu,  
         pw = pw,  
         up = up,  
         wp = wp,
         pp = pp)  

    # get grids
    gp = Grid("../meshes/$geo/mesh$nref.h5", order-2)
    gw = Grid("../meshes/$geo/mesh$nref.h5", order-1)
    gu = Grid("../meshes/$geo/mesh$nref.h5", order)
    g = (p = gp, u = gu, w = gw)

    # get Jacobians
    g1 = Grid("../meshes/$geo/mesh$nref.h5", 1)
    J = Jacobians(g1)

    # top and bottom edges
    ebotw, etopw = get_sides(gw)
    ebotu, etopu = get_sides(gu)
    e = (botw = ebotw, topw = etopw, 
         botu = ebotu, topu = etopu)

    # mesh resolution 
    hp = 1/sqrt(gp.np)
    huᶻ = 1/sqrt(gw.np)
    huˣ = 1/sqrt(gu.np)

    # # exact solution
    # xu = gu.p[:, 1] 
    # zu = gu.p[:, 2] 
    # xw = gw.p[:, 1] 
    # zw = gw.p[:, 2] 
    # xp = gp.p[:, 1] 
    # zp = gp.p[:, 2] 
    # uˣa = @.  cos(π*xu/2)*sin(π*zu/2)
    # uᶻa = @. -sin(π*xw/2)*cos(π*zw/2)
    # pa = @. sin(xp*zp)*exp(zp) 
    # fˣ = @. zu*cos(xu*zu)*exp(zu) + π^2/4*cos(π*xu/2)*sin(π*zu/2)
    # fᶻ = @. xw*cos(xw*zw)*exp(zw) + sin(xw*zw)*exp(zw)

    # # forcing and dirichlet for solver
    # f = (x = fˣ, z = fᶻ)
    # u₀ = (botw = uᶻa[ebotw], topw = uᶻa[etopw],
    #       botu = uˣa[ebotu], topu = uˣa[etopu])
    # p₀ = pa[1]

    # more realistic example 
    x = gw.p[:, 1] 
    z = gw.p[:, 2] 
    H_func(x) = sqrt(2 - x^2) - 1
    # H_func(x) = 1 - x^2
    H = H_func.(x)
    δ = 0.2
    fᶻ = @. z + δ*H*exp(-(z/H + 1)/δ)
    fᶻ[H .== 0] .= 0
    fˣ = zeros(gu.np)
    f = (x = fˣ, z = fᶻ)
    u₀ = (botw = zeros(size(ebotw)), topw = zeros(size(etopw)),
          botu = zeros(size(ebotu)), topu = zeros(size(etopu)))
    p₀ = 0

    # solve stokes_hydro problem
    # uˣ, uᶻ, p = solve_stokes_hydro(g, s, J, e, f, u₀, p₀)
    uˣ, uᶻ, p = solve_stokes_hydro(g, s, J, e, f, u₀, p₀; diri_mask=(true, true, false, true))

    if plot
        # quickplot(g.u, uˣ, L"u^x", "images/ux.png")
        # quickplot(g.w, uᶻ, L"u^z", "images/uz.png")
        # quickplot(g.w, p, L"p", "images/p.png")
        # quickplot(g.u, uˣa, L"u^x_a", "images/uxa.png")
        # quickplot(g.w, uᶻa, L"u^z_a", "images/uza.png")
        # quickplot(g.w, pa, L"p_a", "images/pa.png")
        quickplot(-1:0.01:1, H_func.(-1:0.01:1), g.w, f.z, g.u, uˣ, L"u^x", "images/ux.png")
        quickplot(-1:0.01:1, H_func.(-1:0.01:1), g.w, f.z, g.w, uᶻ, L"u^z", "images/uz.png")
        quickplot(-1:0.01:1, H_func.(-1:0.01:1), g.w, f.z, g.w, p, L"p", "images/p.png")
    end

    # error
    # err_uˣ = L2norm(g.u, s.uu, J, uˣ - uˣa)
    # err_uᶻ = L2norm(g.w, s.ww, J, uᶻ - uᶻa)
    # err_p  = L2norm(g.p, s.pp, J, p - pa)
    err_uˣ = 0
    err_uᶻ = 0
    err_p  = 0
    return huˣ, huᶻ, hp, err_uˣ, err_uᶻ, err_p
end

"""
    stokes_hydro_conv(nrefs)
"""
function stokes_hydro_conv(nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"$L_2$ Error") 
    for i in eachindex(nrefs)
        println(nrefs[i])
        huˣ, huᶻ, hp, err_uˣ, err_uᶻ, err_p = stokes_hydro_res(nrefs[i])
        ax.loglog(huˣ, err_uˣ, c="tab:blue", "o")
        ax.loglog(huᶻ, err_uᶻ, c="tab:orange", "o")
        ax.loglog(hp, err_p, c="tab:green", "o")
    end
    hmin = 2e-3
    hmax = 1e-1
    err_min = 2e-7
    err_max = 5e-2
    ax.loglog([1e-1, 1e-2], [5e-3, 5e-3*(1e-1)^2], "k-")
    legend_elements = [
        Line2D([0], [0], color="w", markerfacecolor="tab:blue", marker="o", label=L"u^x"),
        Line2D([0], [0], color="w", markerfacecolor="tab:orange", marker="o", label=L"u^z"),
        Line2D([0], [0], color="w", markerfacecolor="tab:green", marker="o", label=L"p"),
        Line2D([0], [0], color="k", label=L"O(h^2)")
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlim(0.5*hmin, 2*hmax)
    ax.set_ylim(0.5*err_min, 2*err_max)
    savefig("images/stokes_hydro.png")
    println("images/stokes_hydro.png")
    plt.close()
end

stokes_hydro_res(0; plot=true)
# stokes_hydro_conv(0:5)

println("Done.")