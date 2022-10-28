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
    ux, uz, p = solve_stokes_hydro(g, s, J, b, e, f, u₀)

stokes_hydro problem:
    -∂zz(ux) + ∂x(p) = fˣ,
               ∂z(p) = fᶻ,
     ∂x(ux) + ∂z(uz) = 0, 
with extra condition
    ∫ p dx dz = 0,
and Dirichlet boundary conditions on u.
Weak form:
    ∫ [ ∂z(ux)∂z(vx) - p∂x(vx) 
      - p∂z(vz)
      + q∂x(ux) + q∂z(uz)
      ] dx dz
    = ∫ [fˣvx + fᶻvz] dx dz,
for all 
    ux, vx ∈ Pᵢ,
    uz, vz ∈ Pⱼ, 
    p, q ∈ Pₖ
where j = i-1, k = i-2, and Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_hydro(ux, uz, p, fx, fz, J, s, e, u₀, p₀; diri_mask=(true, true, true, true))
    # indices
    uxmap = 1:ux.g.np
    uzmap = uxmap[end] .+ (1:uz.g.np)
    pmap  = uzmap[end] .+ (1:p.g.np)
    N = pmap[end]
    println("N = $N")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:ux.g1.nt
        # ∂z(ux)∂z(vx)
        Kᵏ = abs(J.J[k])*(s.uu.φξφξ*J.ξy[k]^2 + s.uu.φξφη*J.ξy[k]*J.ηy[k] + s.uu.φηφξ*J.ηy[k]*J.ξy[k] + s.uu.φηφη*J.ηy[k]^2)

        # p*∂x(vx) 
        Cx_puᵏ = abs(J.J[k])*(s.pu.φφξ*J.ξx[k] + s.pu.φφη*J.ηx[k])
        # p*∂z(vz)
        Cz_pwᵏ = abs(J.J[k])*(s.pw.φφξ*J.ξy[k] + s.pw.φφη*J.ηy[k])
        # q*∂x(ux) 
        Cx_upᵏ = abs(J.J[k])*(s.up.φξφ*J.ξx[k] + s.up.φηφ*J.ηx[k])
        # q*∂z(uz)
        Cz_wpᵏ = abs(J.J[k])*(s.wp.φξφ*J.ξy[k] + s.wp.φηφ*J.ηy[k])

        # δ*q*p
        Mppᵏ = abs(J.J[k])*s.pp.φφ

        # fˣ*vx
        rxᵏ = abs(J.J[k])*s.uu.φφ*fx.values[ux.g.t[k, :]]
        # fᶻ*vz
        rzᵏ = abs(J.J[k])*s.ww.φφ*fz.values[uz.g.t[k, :]]

        # ux*vx
        for i=1:ux.g.nn, j=1:ux.g.nn
            # x-mom: ∂z(ux)∂z(vx)
            push!(A, (uxmap[ux.g.t[k, i]], uxmap[ux.g.t[k, j]], Kᵏ[i, j]))
        end
        # p*vx
        for i=1:ux.g.nn, j=1:p.g.nn
            # x-mom: -p*∂x(vx)
            push!(A, (uxmap[ux.g.t[k, i]], pmap[p.g.t[k, j]], -Cx_puᵏ[i, j]))
        end
        # ux*q
        for i=1:p.g.nn, j=1:ux.g.nn
            # cont: ∂x(ux)*q
            push!(A, (pmap[p.g.t[k, i]], uxmap[ux.g.t[k, j]], Cx_upᵏ[i, j]))
        end
        # p*vz
        for i=1:uz.g.nn, j=1:p.g.nn
            # z-mom: -p*∂z(vz)
            push!(A, (uzmap[uz.g.t[k, i]], pmap[p.g.t[k, j]], -Cz_pwᵏ[i, j]))
        end
        # uz*q
        for i=1:p.g.nn, j=1:uz.g.nn
            # cont: ∂z(uz)*q
            push!(A, (pmap[p.g.t[k, i]], uzmap[uz.g.t[k, j]], Cz_wpᵏ[i, j]))
        end
        # # p*p
        # for i=1:p.g.nn, j=1:p.g.nn
        #     # pressure condition: δ*q*p
        #     push!(A, (pmap[p.g.t[k, i]], pmap[p.g.t[k, j]], 1e-7*Mppᵏ[i, j]))
        # end
        # f
        r[uxmap[ux.g.t[k, :]]] .+= rxᵏ
        r[uzmap[uz.g.t[k, :]]] .+= rzᵏ
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet condition on bottom and top (replace mom eqtns)
    if diri_mask[1]
        A, r = add_dirichlet(A, r, uxmap[e.botu], u₀.botu)
    end
    if diri_mask[2]
        A, r = add_dirichlet(A, r, uzmap[e.botw], u₀.botw)
    end
    if diri_mask[3]
        A, r = add_dirichlet(A, r, uxmap[e.topu], u₀.topu)
    end
    if diri_mask[4]
        A, r = add_dirichlet(A, r, uzmap[e.topw], u₀.topw)
    end
    # pressure condition
    A, r = apply_constraint(A, r, pmap[1], pmap[1], p₀)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    if N < 1000
        M = Matrix(A)
        fig, ax = subplots(1)
        ax.imshow(abs.(M) .== 0, cmap="binary_r")
        ax.spines["left"].set_visible(false)
        ax.spines["bottom"].set_visible(false)
        savefig("images/A.png")
        println("images/A.png")
        plt.close()
        println("Condition number: ", cond(M))
        println("rank(A) = ", rank(M))
        println("A is sym: ", issymmetric(M))
    end

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    println(@sprintf("%.1f s", time() - t₀))

    # reshape to get u and p
    ux.values[:] = sol[uxmap]
    uz.values[:] = sol[uzmap]
    p.values[:] = sol[pmap]
    return ux, uz, p
end

"""
    ux, uz, p = stokes_hydro_res(nref)
    hs, errs  = stokes_hydro_res(nref)
"""
function stokes_hydro_res(nref, geo; plot=false, exact=false)
    # order of polynomials
    order = 2

    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"
    gp = FEGrid(gfile, order-2)
    gw = FEGrid(gfile, order-1)
    gu = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    ww = ShapeFunctionIntegrals(gw.s, gw.s)
    pu = ShapeFunctionIntegrals(gp.s, gu.s)
    pw = ShapeFunctionIntegrals(gp.s, gw.s)
    up = ShapeFunctionIntegrals(gu.s, gp.s)
    wp = ShapeFunctionIntegrals(gw.s, gp.s)
    pp = ShapeFunctionIntegrals(gp.s, gp.s)
    s = (uu = uu,
         ww = ww, 
         pu = pu,  
         pw = pw,  
         up = up,  
         wp = wp,
         pp = pp)  

    # get Jacobians
    J = Jacobians(g1)

    # top and bottom edges
    ebotw, etopw = get_sides(gw)
    ebotu, etopu = get_sides(gu)
    e = (botw = ebotw, topw = etopw, 
         botu = ebotu, topu = etopu)

    if exact 
        # mesh resolution 
        hp = 1/sqrt(gp.np)
        huz = 1/sqrt(gw.np)
        hux = 1/sqrt(gu.np)

        # exact solution
        xu = gu.p[:, 1] 
        zu = gu.p[:, 2] 
        xw = gw.p[:, 1] 
        zw = gw.p[:, 2] 
        xp = gp.p[:, 1] 
        zp = gp.p[:, 2] 
        uxa = @.  cos(π*xu/2)*sin(π*zu/2)
        uza = @. -sin(π*xw/2)*cos(π*zw/2)
        pa = @. sin(xp*zp)*exp(zp) 
        fx = @. zu*cos(xu*zu)*exp(zu) + π^2/4*cos(π*xu/2)*sin(π*zu/2)
        fz = @. xw*cos(xw*zw)*exp(zw) + sin(xw*zw)*exp(zw)
        u₀ = (botw = uza[ebotw], topw = uza[etopw],
              botu = uxa[ebotu], topu = uxa[etopu])
        p₀ = pa[1]
        diri_mask = (true, true, true, true)
    else
        # forcing
        function H(x)
            if geo == "gmsh_tri"
                return 1 - abs(x)
            else
                return sqrt(2 - x^2) - 1
            end
        end
        fx = zeros(gu.np)
        x = gw.p[:, 1] 
        z = gw.p[:, 2] 
        δ = 0.1
        # fz = @. z + δ*exp(-(z + H)/δ)
        # fz = z
        fz = @. δ*exp(-(z + H(x))/δ)
        u₀ = (botw = zeros(size(ebotw)), topw = zeros(size(etopw)),
            botu = zeros(size(ebotu)), topu = zeros(size(etopu)))
        p₀ = 0
        diri_mask = (true, true, false, true)
    end

    # initialize FE fields
    ux = FEField(order,    zeros(gu.np), gu, g1)
    uz = FEField(order-1,  zeros(gw.np), gw, g1)
    p  = FEField(order-2,  zeros(gp.np), gp, g1)
    fx  = FEField(order,   fx,           gu, g1)
    fz  = FEField(order-1, fz,           gw, g1)

    # solve stokes_hydro problem
    ux, uz, p = solve_stokes_hydro(ux, uz, p, fx, fz, J, s, e, u₀, p₀; diri_mask=diri_mask)

    if plot
        quickplot(fz, ux, L"u^x", "images/ux.png")
        quickplot(fz, uz, L"u^z", "images/uz.png")
        quickplot(fz, p, L"p", "images/p.png")
        plot_profile(ux, 0.5, -H(0.5)+1e-5:1e-3:0, L"u^x", L"z", "images/ux_profile.png")
        plot_profile(uz, 0.5, -H(0.5)+1e-5:1e-3:0, L"u^z", L"z", "images/uz_profile.png")
        plot_profile(p, 0.5, -H(0.5)+1e-5:1e-3:0, L"p", L"z", "images/p_profile.png")
    end

    if exact
        err_ux = L2norm(ux.g, s.uu, J, ux.values - uxa)
        err_uz = L2norm(uz.g, s.ww, J, uz.values - uza)
        err_p  = L2norm(p.g, s.pp, J, p.values - pa)
        return hux, huz, hp, err_ux, err_uz, err_p
    else
        return ux, uz, p
    end
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
        hux, huz, hp, err_ux, err_uz, err_p = stokes_hydro_res(nrefs[i], "jc"; exact=true)
        ax.loglog(hux, err_ux, c="tab:blue", "o")
        ax.loglog(huz, err_uz, c="tab:orange", "o")
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

# stokes_hydro_res(3, "jc"; plot=true)
# stokes_hydro_res(4, "gmsh"; plot=true)
# stokes_hydro_res(5, "gmsh_tri"; plot=true)
stokes_hydro_res(0, "valign"; plot=true)

# stokes_hydro_conv(0:5)

println("Done.")
