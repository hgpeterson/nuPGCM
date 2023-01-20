using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using Printf

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, f, diri, J, s, e, ε²)

PG Inversion:
    -ε²∂zz(ωx) - ωy = f₁, 
    -ε²∂zz(ωy) + ωx = f₂,
       ∂zz(χx) + ωx = f₃,
       ∂zz(χy) + ωy = f₄,
with boundary conditions 
    ωx, ωy, χx, χy dirichlet at z = 0,
    ∂z(χx) = ∂z(χy) = 0  at  z = -H,
    χx, χy dirichlet at z = -H.
"""
function solve_pg_vort_2D(ωx, ωy, χx, χy, f, diri, J, s, e, ε²)
    # unpack grids
    g1 = ωx.g1
    g = ωx.g
    # indices
    ωxmap = 1:g.np
    ωymap = (g.np+1):2*g.np
    χxmap = (2*g.np+1):3*g.np
    χymap = (3*g.np+1):4*g.np
    N = 4*g.np

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # stiffness and mass matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # RHSs
        r[ωxmap[g.t[k, :]]] += M*f.f1.values[g.t[k, :]]
        r[ωymap[g.t[k, :]]] += M*f.f2.values[g.t[k, :]]
        r[χxmap[g.t[k, :]]] += M*f.f3.values[g.t[k, :]]
        r[χymap[g.t[k, :]]] += M*f.f4.values[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            # indices
            ωxi = ωxmap[g.t[k, :]]
            ωyi = ωymap[g.t[k, :]]
            χxi = χxmap[g.t[k, :]]
            χyi = χymap[g.t[k, :]]

            # -ε²*∂zz(ωx)
            push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
            # -ωy
            push!(A, (ωxi[i], ωyi[j], -M[i, j]))

            # -ε²*∂zz(ωy)
            push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
            # +ωx
            push!(A, (ωyi[i], ωxi[j], M[i, j]))

            # +∂zz(χx)
            push!(A, (χxi[i], χxi[j], -K[i, j]))
            # +ωx
            push!(A, (χxi[i], ωxi[j], M[i, j]))

            # +∂zz(χy)
            push!(A, (χyi[i], χyi[j], -K[i, j]))
            # +ωy
            push!(A, (χyi[i], ωyi[j], M[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # bottom: dirichlet
    A, r = add_dirichlet(A, r, ωxmap[e.bot], χxmap[e.bot], diri.χx_bot) 
    A, r = add_dirichlet(A, r, ωymap[e.bot], χymap[e.bot], diri.χy_bot)
    # A, r = add_dirichlet(A, r, ωxmap[e.bot], diri.ωx_bot)
    # A, r = add_dirichlet(A, r, ωymap[e.bot], diri.ωy_bot)
    # A, r = add_dirichlet(A, r, χxmap[e.bot], diri.χx_bot)
    # A, r = add_dirichlet(A, r, χymap[e.bot], diri.χy_bot)

    # sfc: dirichlet 
    A, r = add_dirichlet(A, r, ωxmap[e.sfc], diri.ωx_sfc)
    A, r = add_dirichlet(A, r, ωymap[e.sfc], diri.ωy_sfc)
    A, r = add_dirichlet(A, r, χxmap[e.sfc], diri.χx_sfc)
    A, r = add_dirichlet(A, r, χymap[e.sfc], diri.χy_sfc)

    # # special dirichlet conditions ∂x(ωx) = ∂x(χy) = 0 at z = -H
    # emap, edges, bndix = all_edges(g1.t)
    # w_quad, t_quad = quad_weights_points(2*g.order-1, 1)
    # ps = reference_element_nodes(g.order, g.dim)
    # A[ωxmap[e.bot], :] .= 0
    # r[ωxmap[e.bot]] .= 0
    # A[ωymap[e.bot], :] .= 0
    # r[ωymap[e.bot]] .= 0
    # for k=1:g1.nt, ie=1:3
    #     if emap[k, ie] in bndix # edge `ie` of triangle `k` is on the boundary
    #         # get local indices of each point on edge `ie`:
    #         if g.order == 1
    #             il = [ie, mod1(ie+1, 3)]
    #         elseif g.order == 2
    #             il = [ie, ie+3, mod1(ie+1, 3)]
    #         end
    #         ig = g.t[k, il]
    #         if (ig[1] in e.bot) && (ig[end] in e.bot) # the edge is on the *bottom* boundary
    #             # get global coordinates of end points on edge
    #             p1 = g.p[ig[1], :]
    #             p2 = g.p[ig[end], :]

    #             # get local coordinates on standard triangle of each point on edge
    #             ξ1 = ps[il[1], :]
    #             ξ2 = ps[il[end], :]

    #             # get ∂ξ/∂x and ∂η/∂x
    #             ξx = J.Js[k, 1, 1]
    #             ηx = J.Js[k, 2, 1]

    #             # compute ∫ φᵢ(ξ(t))*∂x(φⱼ(ξ(t)))*||ξ′(t)||*dt for t ∈ [-1, 1] where ξ(-1) = ξ1 and ξ(1) = ξ2
    #             ξ(t) = (ξ2 - ξ1)/2*t + (ξ2 + ξ1)/2
    #             for i=il, j=1:g.nn
    #                 f(t) = φ(g.s, i, ξ(t))*∂φ(g.s, j, 1, ξ(t))*norm(p2 - p1)/(p2[1] - p1[1])/2 # TF ∂ξ
    #                 ∫f = dot(w_quad, f.(t_quad))
    #                 A[ωxmap[g.t[k, i]], ωxmap[g.t[k, j]]] += ∫f

    #                 f1(t) = φ(g.s, i, ξ(t))*(∂φ(g.s, j, 1, ξ(t))*ξx + ∂φ(g.s, j, 2, ξ(t))*ηx)*norm(p2 - p1)/2
    #                 ∫f = dot(w_quad, f1.(t_quad))
    #                 A[ωymap[g.t[k, i]], χymap[g.t[k, j]]] += ∫f
    #             end
    #         end
    #     end
    # end

    # remove zeros
    dropzeros!(A)

    # solve
    sol = A\r

    # reshape to get ω and χ
    ωx.values[:] = sol[ωxmap]
    ωy.values[:] = sol[ωymap]
    χx.values[:] = sol[χxmap]
    χy.values[:] = sol[χymap]
    return ωx, ωy, χx, χy
end

function pg_vort_2D_res(; nref, order, showplots=false)
    # Ekman number
    ε² = 1

    # setup FE grids
    # gfile = "../meshes/gmsh/mesh$nref.h5"
    gfile = "../meshes/valign2D/mesh$nref.h5"
    g  = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # mesh resolution 
    h = 1/sqrt(g.np)

    # sfc and bottom edges
    ebot, esfc = get_sides(g)
    e = (bot=ebot, sfc=esfc) 
    
    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    # constructed solution
    x = g.p[:, 1] 
    z = g.p[:, 2] 
    H = @. 1 - x^2
    # H = @. sqrt(2 - x^2) - 1
    ωx_a = @. x*exp(x*z)
    ωy_a = @. x*exp(x*z)
    χx_a = @. -(1 - H + exp(z)*(-1 + H + z))*sin(x)
    χy_a = @. -(1 - H + exp(z)*(-1 + H + z))*cos(x)
    diri = (ωx_bot=ωx_a[e.bot], ωx_sfc=ωx_a[e.sfc],
            ωy_bot=ωy_a[e.bot], ωy_sfc=ωy_a[e.sfc],
            χx_bot=χx_a[e.bot], χx_sfc=χx_a[e.sfc],
            χy_bot=χy_a[e.bot], χy_sfc=χy_a[e.sfc],
           )
    ωx_a = FEField(ωx_a, g, g1)
    ωy_a = FEField(ωy_a, g, g1)
    χx_a = FEField(χx_a, g, g1)
    χy_a = FEField(χy_a, g, g1)

    # forcing
    f1 = @. -x*exp(x*z) - ε²*x^3*exp(x*z)
    f2 = @.  x*exp(x*z) - ε²*x^3*exp(x*z)
    f3 = @. x*exp(x*z) + (-2*exp(z) - exp(z)*(-1 + H + z))*sin(x)
    f4 = @. x*exp(x*z) + (-2*exp(z) - exp(z)*(-1 + H + z))*cos(x)
    f1 = FEField(f1, g, g1)
    f2 = FEField(f2, g, g1)
    f3 = FEField(f3, g, g1)
    f4 = FEField(f4, g, g1)
    f = (f1=f1, f2=f2, f3=f3, f4=f4)

    # initialize FE fields
    ωx = FEField(zeros(g.np), g, g1)
    ωy = FEField(zeros(g.np), g, g1)    
    χx = FEField(zeros(g.np), g, g1)
    χy = FEField(zeros(g.np), g, g1)

    # solve 
    ωx, ωy, χx, χy = solve_pg_vort_2D(ωx, ωy, χx, χy, f, diri, J, s, e, ε²)

    if showplots
        quickplot(ωx, L"\omega^x", "images/omegax.png")
        quickplot(ωy, L"\omega^y", "images/omegay.png")
        quickplot(χx, L"\chi^x",   "images/chix.png")
        quickplot(χy, L"\chi^y",   "images/chiy.png")
    end

    err = L2norm(ωx - ωx_a, s, J) +
          L2norm(ωy - ωy_a, s, J) +
          L2norm(χx - χx_a, s, J) +
          L2norm(χy - χy_a, s, J)

    return h, err
end

function pg_vort_2D_conv(; nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"$L^2$ Error")
    for o=1:2
        println("Order ", o)
        h = zeros(size(nrefs, 1))
        err = zeros(size(nrefs, 1))
        for i in eachindex(nrefs)
            println("\tRefinement ", nrefs[i])
            h[i], err[i] = pg_vort_2D_res(nref=nrefs[i], order=o)
        end
        ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^(o)], "k-", alpha=o/3, label=latexstring(L"$h^", o, L"$"))
        ax.loglog(h, err, "o", label="Order $o")
    end
    ax.legend(ncol=2, loc=(0.0, 1.05))
    savefig("images/pg_vort_2D.png")
    println("images/pg_vort_2D.png")
    plt.close()
end

# h, err = pg_vort_2D_res(nref=3, order=2, showplots=true)
pg_vort_2D_conv(nrefs=0:3)

println("Done.")