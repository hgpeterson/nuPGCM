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
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, b, J, s, e, ε²)

PG Inversion:
    -ε²∂zz(ωx) - ωy = 0, 
    -ε²∂zz(ωy) + ωx = -∂x(b),
       ∂zz(χx) + ωx = 0,
       ∂zz(χy) + ωy = 0,
with boundary conditions 
    χx = χy = ωx = ωy = 0  at  z = 0,
      ∂z(χx) = ∂z(χy) = 0  at  z = -H,
              χy = ωx = 0  at  z = -H. (*)
(*) should actually have ∂x(ωx) = ∂x(χy) = 0 at z = -H.
"""
function solve_pg_vort(ωx, ωy, χx, χy, b, J, s, e, ε²)
    # unpack grids
    g1 = ωx.g1
    g = ωx.g
    # indices
    ωxmap = 1:g.np
    ωymap = (g.np+1):2*g.np
    χxmap = (2*g.np+1):3*g.np
    χymap = (3*g.np+1):4*g.np
    N = 4*g.np
    println("N = $N")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # matrices
        K = abs(J.J[k])*(s.φξφξ*J.ξy[k]^2 + s.φξφη*J.ξy[k]*J.ηy[k] + s.φηφξ*J.ηy[k]*J.ξy[k] + s.φηφη*J.ηy[k]^2)
        Cx = abs(J.J[k])*(s.φξφ*J.ξx[k] + s.φηφ*J.ηx[k])
        M = abs(J.J[k])*s.φφ

        # -∂x(b)
        r[ωymap[g.t[k, :]]] -= Cx*b.values[g.t[k, :]]

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

    # top: dirichlet 
    A, r = add_dirichlet(A, r, ωxmap[e.top], 0)
    A, r = add_dirichlet(A, r, ωymap[e.top], 0)
    A, r = add_dirichlet(A, r, χxmap[e.top], 0)
    A, r = add_dirichlet(A, r, χymap[e.top], 0)

    # special dirichlet conditions ∂x(ωx) = ∂x(χy) = 0 at z = -H
    edges, boundary_indices, emap = all_edges(g1.t)
    w, t = quad_weights_points(2*g.order-1, 1)
    ps = standard_element_nodes(g.order)
    A[ωxmap[e.bot], :] .= 0
    r[ωxmap[e.bot]] .= 0
    A[ωymap[e.bot], :] .= 0
    r[ωymap[e.bot]] .= 0
    for k=1:g1.nt
        for ie=1:3
            if emap[k, ie] in boundary_indices # edge `ie` of triangle `k` is on the boundary
                # get local indices of each point on edge `ie`:
                il = [ie, ie+3, mod1(ie+1, 3)]
                ig = g.t[k, il]
                if (ig[1] in e.bot) && (ig[3] in e.bot) # the edge is on the *bottom* boundary
                    # get global coordinates of end points on edge
                    p1 = g.p[ig[1], :]
                    p3 = g.p[ig[3], :]

                    # get local coordinates on standard triangle of each point on edge
                    ξ1 = ps[il[1], :]
                    ξ3 = ps[il[3], :]

                    # compute ∫ φᵢ(ξ(t))*∂x(φⱼ(ξ(t)))*||ξ′(t)||*dt for t ∈ [-1, 1] where ξ(-1) = ξ1 and ξ(1) = ξ3
                    ξ(t) = (ξ3 - ξ1)/2*t + (ξ3 + ξ1)/2
                    for i=il, j=1:g.nn
                        f(t) = φ(g.s, i, ξ(t))*φξ(g.s, j, ξ(t))*norm((ξ3 - ξ1)/2)*norm(p3 - p1)
                        ∫f = dot(w, f.(t))
                        A[ωxmap[g.t[k, i]], ωxmap[g.t[k, j]]] += ∫f

                        # f1(t) = φ(g.s, i, ξ(t))*φ(g.s, j, ξ(t))*norm((ξ3 - ξ1)/2)*norm(p3 - p1)
                        # f1(t) = φ(g.s, i, ξ(t))*φξ(g.s, j, ξ(t))*norm((ξ3 - ξ1)/2)*norm(p3 - p1)
                        f1(t) = φ(g.s, i, ξ(t))*(φξ(g.s, j, ξ(t))*J.ξx[k] + φη(g.s, j, ξ(t))*J.ηx[k])*norm((ξ3 - ξ1)/2)*norm(p3 - p1)
                        ∫f1 = dot(w, f1.(t))
                        A[ωymap[g.t[k, i]], χymap[g.t[k, j]]] += ∫f1
                    end
                end
            end
        end
    end

    # # if we don't do ∂x(ωx) = 0
    # A, r = add_dirichlet(A, r, ωxmap[e.bot], 0) 

    # # if we don't do ∂x(χy) = 0
    # A, r = add_dirichlet(A, r, ωymap[e.bot], χymap[e.bot], 0) # need to apply this on ωy since χy is full

    # corners: dirichlet 
    A, r = add_dirichlet(A, r, ωxmap[e.bot[[1, end]]], 0)
    A, r = add_dirichlet(A, r, ωymap[e.bot[[1, end]]], 0)
    A, r = add_dirichlet(A, r, χxmap[e.bot[[1, end]]], 0)
    A, r = add_dirichlet(A, r, χymap[e.bot[[1, end]]], 0)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    println("rank(A): ", rank(A))

    # null = nullspace(Matrix(A))
    # ωx.values[:] = null[ωxmap]
    # ωy.values[:] = null[ωymap]
    # χx.values[:] = null[χxmap]
    # χy.values[:] = null[χymap]
    # return ωx, ωy, χx, χy

    # solve
    print("Solving... ")
    t₀ = time()
    sol = A\r
    println(@sprintf("%.1f s", time() - t₀))

    # reshape to get ω and χ
    ωx.values[:] = sol[ωxmap]
    ωy.values[:] = sol[ωymap]
    χx.values[:] = sol[χxmap]
    χy.values[:] = sol[χymap]
    return ωx, ωy, χx, χy
end

function pg_vort_res(geo, nref; showplots=false)
    # order of polynomials
    order = 2

    # Ekman number
    # ε² = 1e-5
    # ε² = 1e-4
    # ε² = 1e-3
    # ε² = 1e-2
    ε² = 1e-1
    # ε² = 1

    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"
    g  = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    println(@sprintf("q⁻¹ = %1.1e", sqrt(2*ε²)))
    println(@sprintf("h   = %1.1e", 1/sqrt(g1.np)))

    # top and bottom edges
    ebot, etop = get_sides(g)
    e = (bot = ebot, top = etop) 

    # forcing
    function H(x)
        if geo == "gmsh_tri"
            return 1 - abs(x)
        else
            return sqrt(2 - x^2) - 1
        end
    end
    x = g.p[:, 1] 
    z = g.p[:, 2] 
    δ = 0.1
    b = @. z + δ*exp(-(z + H(x))/δ)

    # initialize FE fields
    ωx = FEField(zeros(g.np), g, g1)
    ωy = FEField(zeros(g.np), g, g1)
    χx = FEField(zeros(g.np), g, g1)
    χy = FEField(zeros(g.np), g, g1)
    b  = FEField(b,           g, g1)

    # solve 
    ωx, ωy, χx, χy = solve_pg_vort(ωx, ωy, χx, χy, b, J, s, e, ε²)

    if showplots
        quickplot(ωx, L"\omega^x", "images/omegax.png")
        quickplot(ωy, L"\omega^y", "images/omegay.png")
        quickplot(χx, L"\chi^x",   "images/chix.png")
        quickplot(χy, L"\chi^y",   "images/chiy.png")
        plot_profile(ωx, 0.5, -H(0.5):1e-3:0, L"$\omega^x$ at $x = 0.5$", L"z", "images/omegax_profile.png")
        plot_profile(ωy, 0.5, -H(0.5):1e-3:0, L"$\omega^y$ at $x = 0.5$", L"z", "images/omegay_profile.png")
        plot_profile(χx, 0.5, -H(0.5):1e-3:0, L"$\chi^x$ at $x = 0.5$",   L"z", "images/chix_profile.png")
        plot_profile(χy, 0.5, -H(0.5):1e-3:0, L"$\chi^y$ at $x = 0.5$",   L"z", "images/chiy_profile.png")
    end

    return ωx, ωy, χx, χy
end

"""
    ux, uy, uz = get_velocities(χx, χy)

Solve the equations
    ux = -∂z(χy)
    uy = ∂z(χx)
    uz = ∂x(χy)
With b.c. 
       ux = uy = uz = 0  at  z = -H,
    ∂z(ux) = ∂z(uy) = 0  at  z = 0,
                 uz = 0  at  z = 0.
"""
function get_velocities(χx, χy; showplots=false)
    # unpack grids
    g1 = χx.g1
    g2 = χx.g

    # set up order 1 velocity fields
    ux = FEField(zeros(g1.np), g1, g1)
    uy = FEField(zeros(g1.np), g1, g1)
    uz = FEField(zeros(g1.np), g1, g1)

    # Jacobians
    J = Jacobians(g1)   

    # integrals
    s1 = ShapeFunctionIntegrals(g1.s, g1.s)
    s21 = ShapeFunctionIntegrals(g2.s, g1.s)

    # stamp system
    M = Tuple{Int64,Int64,Float64}[]
    Cx = Tuple{Int64,Int64,Float64}[]
    Cz = Tuple{Int64,Int64,Float64}[]
    for k=1:g1.nt
        # matrices
        Mᵏ = abs(J.J[k])*s1.φφ
        Cxᵏ = abs(J.J[k])*(s21.φξφ*J.ξx[k] + s21.φηφ*J.ηx[k])
        Czᵏ = abs(J.J[k])*(s21.φξφ*J.ξy[k] + s21.φηφ*J.ηy[k])
        for i=1:g1.nn, j=1:g2.nn
            push!(Cx, (g1.t[k, i], g2.t[k, j], Cxᵏ[i, j]))
            push!(Cz, (g1.t[k, i], g2.t[k, j], Czᵏ[i, j]))
        end
        for i=1:g1.nn, j=1:g1.nn
            push!(M, (g1.t[k, i], g1.t[k, j], Mᵏ[i, j]))
        end
    end

    # make CSC matrices
    M = sparse((x -> x[1]).(M), (x -> x[2]).(M), (x -> x[3]).(M), g1.np, g1.np)
    Cx = sparse((x -> x[1]).(Cx), (x -> x[2]).(Cx), (x -> x[3]).(Cx), g1.np, g2.np)
    Cz = sparse((x -> x[1]).(Cz), (x -> x[2]).(Cz), (x -> x[3]).(Cz), g1.np, g2.np)

    # ux = -∂z(χy)
    ux.values[:] = -M\(Cz*χy.values)

    # uy = ∂z(χx)
    uy.values[:] = M\(Cz*χx.values)

    # uz = ∂x(χy)
    uz.values[:] = M\(Cx*χy.values)

    if showplots
        quickplot(ux, L"u^x", "images/ux.png")
        quickplot(uy, L"u^y", "images/uy.png")
        quickplot(uz, L"u^z", "images/uz.png")
        H(x) = sqrt(2 - x^2) - 1
        plot_profile(ux, 0.5, -H(0.5):1e-3:0, L"$u^x$ at $x = 0.5$", L"z", "images/ux_profile.png")
        plot_profile(uy, 0.5, -H(0.5):1e-3:0, L"$u^y$ at $x = 0.5$", L"z", "images/uy_profile.png")
        plot_profile(uz, 0.5, -H(0.5):1e-3:0, L"$u^z$ at $x = 0.5$", L"z", "images/uz_profile.png")
    end

    return ux, uy, uz
end

ωx, ωy, χx, χy = pg_vort_res("gmsh", 3; showplots=true)
# ωx, ωy, χx, χy = pg_vort_res("", 0; showplots=true)
# ωx, ωy, χx, χy = pg_vort_res("valign", 0; showplots=true)

# ux, uy, uz = get_velocities(χx, χy; showplots=true)

println("Done.")