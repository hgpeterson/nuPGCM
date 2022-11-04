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
           -∂zzz(ux) = ∂x(fᶻ) + ∂z(fˣ),
     ∂x(ux) + ∂z(uz) = 0, 
with Dirichlet boundary conditions on u.
Weak form:
    ∫ ( ∂zz(ux)*∂z(vx) + [∂x(ux) + ∂z(uz)]*vz ) dx dz
    = ∫ [∂x(fᶻ) + ∂z(fˣ)]*vx dx dz,
for all 
    ux, vx ∈ Pᵢ,
    uz, vz ∈ Pⱼ, 
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes_hydro(ux, uz, fx, fz, J, s, e, u₀; diri_mask=(true, true, true, true))
    # indices
    uxmap = 1:ux.g.np
    uzmap = uxmap[end] .+ (1:uz.g.np)
    N = uzmap[end]
    println("N = $N")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:ux.g1.nt
        # ∂zz(ux)∂z(vx)
        Kᵏ = abs(J.J[k])*(s.uu.φξξφξ*J.ξy[k]^3 + 2*s.uu.φξηφξ*J.ξy[k]^2*J.ηy[k] + s.uu.φηηφξ*J.ξy[k]*J.ηy[k]^2 + 
                          s.uu.φξξφη*J.ξy[k]^2*J.ηy[k] + 2*s.uu.φξηφη*J.ξy[k]*J.ηy[k]^2 + s.uu.φηηφη*J.ηy[k]^3) 

        # ∂x(ux)*vz
        Cxᵏ = abs(J.J[k])*(s.uw.φξφ*J.ξx[k] + s.uw.φηφ*J.ηx[k])
        # ∂z(uz)*vz
        Czᵏ = abs(J.J[k])*(s.ww.φξφ*J.ξy[k] + s.ww.φηφ*J.ηy[k])

        # ∂x(fᶻ)*vx + ∂z(fˣ)*vx
        rᵏ = abs(J.J[k])*(s.uu.φξφ*J.ξy[k] + s.uu.φηφ*J.ηy[k])*fx.values[fx.g.t[k, :]] +
             abs(J.J[k])*(s.uu.φξφ*J.ξx[k] + s.uu.φηφ*J.ηx[k])*fz.values[fz.g.t[k, :]]

        # x-mom: ∂zz(ux)∂z(vx)
        for i=1:ux.g.nn, j=1:ux.g.nn
            push!(A, (uxmap[ux.g.t[k, i]], uxmap[ux.g.t[k, j]], Kᵏ[i, j]))
        end
        # cont: ∂x(ux)*vz
        for i=1:uz.g.nn, j=1:ux.g.nn
            push!(A, (uzmap[uz.g.t[k, i]], uxmap[ux.g.t[k, j]], Cxᵏ[i, j]))
        end
        # cont: ∂z(uz)*vz
        for i=1:uz.g.nn, j=1:uz.g.nn
            push!(A, (uzmap[uz.g.t[k, i]], uzmap[uz.g.t[k, j]], Czᵏ[i, j]))
        end
        # f
        r[uxmap[ux.g.t[k, :]]] += rᵏ
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
    return ux, uz
end

"""
    ux, uz = stokes_hydro_res(nref)
    hs, errs  = stokes_hydro_res(nref)
"""
function stokes_hydro_res(nref, geo; showplots=false, exact=false)
    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"

    gu = FEGrid(gfile, 2)
    gw = FEGrid(gfile, 1)
    g1 = FEGrid(gfile, 1)

    println("Nu = ", gu.np)
    println("Nw = ", gw.np)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    ww = ShapeFunctionIntegrals(gw.s, gw.s)
    uw = ShapeFunctionIntegrals(gu.s, gw.s)
    s = (uu=uu, ww=ww, uw=uw)  

    # get Jacobians
    J = Jacobians(g1)

    # top and bottom edges
    ebotw, etopw = get_sides(gw)
    ebotu, etopu = get_sides(gu)
    e = (botw=ebotw, topw=etopw, botu=ebotu, topu=etopu)

    if exact 
        # mesh resolution 
        huz = 1/sqrt(gw.np)
        hux = 1/sqrt(gu.np)

        # exact solution
        xu = gu.p[:, 1] 
        zu = gu.p[:, 2] 
        xw = gw.p[:, 1] 
        zw = gw.p[:, 2] 
        uxa = @.  cos(π*xu/2)*sin(π*zu/2)
        uza = @. -sin(π*xw/2)*cos(π*zw/2)
        fx = @. zu*cos(xu*zu)*exp(zu) + π^2/4*cos(π*xu/2)*sin(π*zu/2)
        fz = @. xw*cos(xw*zw)*exp(zw) + sin(xw*zw)*exp(zw)
        u₀ = (botw=uza[ebotw], topw=uza[etopw],
              botu=uxa[ebotu], topu=uxa[etopu])
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
        x = gu.p[:, 1] 
        z = gu.p[:, 2] 
        δ = 0.1
        # fz = @. z + δ*exp(-(z + H)/δ)
        # fz = z
        fz = @. δ*exp(-(z + H(x))/δ)
        u₀ = (botw=zeros(size(ebotw)), topw=zeros(size(etopw)),
              botu=zeros(size(ebotu)), topu=zeros(size(etopu)))
        # diri_mask = (true, true, false, true)
        diri_mask = (true, true, true, true)
    end

    # initialize FE fields
    ux  = FEField(zeros(gu.np), gu, g1)
    uz  = FEField(zeros(gw.np), gw, g1)
    fx  = FEField(fx,           gu, g1)
    fz  = FEField(fz,           gu, g1)

    # solve stokes_hydro problem
    ux, uz = solve_stokes_hydro(ux, uz, fx, fz, J, s, e, u₀; diri_mask=diri_mask)

    if showplots
        quickplot(ux, L"u^x", "images/ux.png")
        quickplot(uz, L"u^z", "images/uz.png")
        plot_profile(ux, 0.5, -H(0.5):1e-3:0, L"$u^x$ at $x = 0.5$", L"z", "images/ux_profile.png")
        plot_profile(uz, 0.5, -H(0.5):1e-3:0, L"$u^z$ at $x = 0.5$", L"z", "images/uz_profile.png")
    end

    if exact
        err_ux = L2norm(ux.g, s.uu, J, ux.values - uxa)
        err_uz = L2norm(uz.g, s.ww, J, uz.values - uza)
        return hux, huz, err_ux, err_uz
    else
        return ux, uz
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
        hux, huz, err_ux, err_uz = stokes_hydro_res(nrefs[i], "jc"; exact=true)
        ax.loglog(hux, err_ux, c="tab:blue", "o")
        ax.loglog(huz, err_uz, c="tab:orange", "o")
    end
    hmin = 2e-3
    hmax = 1e-1
    err_min = 2e-7
    err_max = 5e-2
    ax.loglog([1e-1, 1e-2], [5e-3, 5e-3*(1e-1)^2], "k-")
    legend_elements = [
        Line2D([0], [0], color="w", markerfacecolor="tab:blue", marker="o", label=L"u^x"),
        Line2D([0], [0], color="w", markerfacecolor="tab:orange", marker="o", label=L"u^z"),
        Line2D([0], [0], color="k", label=L"O(h^2)")
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlim(0.5*hmin, 2*hmax)
    ax.set_ylim(0.5*err_min, 2*err_max)
    savefig("images/stokes_hydro.png")
    println("images/stokes_hydro.png")
    plt.close()
end

stokes_hydro_res(1, "jc"; showplots=true)
# stokes_hydro_res(4, "gmsh"; showplots=true)
# stokes_hydro_res(5, "gmsh_tri"; showplots=true)
# stokes_hydro_res(0, "valign"; showplots=true)

# stokes_hydro_conv(0:5)

println("Done.")
