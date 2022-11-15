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
    χ = solve_sf_test(χ, f, J, s, e)

Solve
       -∂zz(χ) = f,
with boundary conditions 
    χ = 0  at  z = 0,
    ∂x(χ) = 0  at  z = -H.
"""
function solve_sf_test(χ, f, J, s, e)
    # unpack grids
    g1 = χ.g1
    g = χ.g
    # indices
    println("N = $(g.np)")

    # stamp system
    print("Building... ")
    t₀ = time()
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(g.np)
    for k=1:g1.nt
        # stiffness and mass matrices
        K = abs(J.J[k])*(s.φξφξ*J.ξy[k]^2 + s.φξφη*J.ξy[k]*J.ηy[k] + s.φηφξ*J.ηy[k]*J.ξy[k] + s.φηφη*J.ηy[k]^2)

        # f
        M = abs(J.J[k])*s.φφ
        r[g.t[k, :]] += M*f.values[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            # -∂zz(χ)
            push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), g.np, g.np)

    # top: dirichlet 
    A, r = add_dirichlet(A, r, e.top, 0)

    # # bot: dirichlet
    # A, r = add_dirichlet(A, r, e.bot, 0)

    # special dirichlet condition ∂x(χ) = 0 at z = -H
    edges, boundary_indices, emap = all_edges(g1.t)
    w_quad, t_quad = quad_weights_points(2*g.order-1, 1)
    ps = standard_element_nodes(g.order)
    A[e.bot, :] .= 0
    r[e.bot] .= 0
    for k=1:g1.nt, ie=1:3
        if emap[k, ie] in boundary_indices # edge `ie` of triangle `k` is on the boundary
            # get local indices of each point on edge `ie`:
            if g.order == 1
                il = [ie, mod1(ie+1, 3)]
            elseif g.order == 2
                il = [ie, ie+3, mod1(ie+1, 3)]
            end
            ig = g.t[k, il]
            if (ig[1] in e.bot) && (ig[end] in e.bot) # the edge is on the *bottom* boundary
                # get global coordinates of end points on edge
                p1 = g.p[ig[1], :]
                p2 = g.p[ig[end], :]

                # get local coordinates on standard triangle of each point on edge
                ξ1 = ps[il[1], :]
                ξ2 = ps[il[end], :]

                # compute ∫ φᵢ(ξ(t))*∂x(φⱼ(ξ(t)))*||ξ′(t)||*dt for t ∈ [-1, 1] where ξ(-1) = ξ1 and ξ(1) = ξ2
                ξ(t) = (ξ2 - ξ1)/2*t + (ξ2 + ξ1)/2
                for i=il, j=1:g.nn
                    f1(t) = φ(g.s, i, ξ(t))*(φξ(g.s, j, ξ(t))*J.ξx[k] + φη(g.s, j, ξ(t))*J.ηx[k])*norm(p2 - p1)/2
                    # f1(t) = φ(g.s, i, ξ(t))*φξ(g.s, j, ξ(t))*norm(p2 - p1)/(p2[1] - p1[1])/2
                    ∫f = dot(w_quad, f1.(t_quad))
                    A[g.t[k, i], g.t[k, j]] += ∫f
                end
            end
        end
    end

    # corners: dirichlet 
    A, r = add_dirichlet(A, r, e.bot[1], 0)
    A, r = add_dirichlet(A, r, e.bot[end], 0)

    # remove zeros
    dropzeros!(A)
    println(@sprintf("%.1f s", time() - t₀))

    R = rank(A)
    println("rank(A): ", R, " = N - ", g.np - R)
    if R < g.np
        if g.np > 2000
            error("🐻")
        end
        χ.values[:] = nullspace(Matrix(A))
        return χ
    end

    # solve
    print("Solving... ")
    t₀ = time()
    χ.values[:] = A\r
    println(@sprintf("%.1f s", time() - t₀))

    return χ
end

function sf_test_res(geo, nref; showplots=false)
    # order of polynomials
    order = 1
    # order = 2

    # setup FE grids
    gfile = "../meshes/$geo/mesh$nref.h5"
    g  = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)
 
    # get Jacobians
    J = Jacobians(g1)   

    # top and bottom edges
    ebot, etop = get_sides(g)
    e = (bot = ebot, top = etop) 

    # forcing
    x = g.p[:, 1] 
    z = g.p[:, 2] 
    f = @. sin(x) + exp(z)

    # initialize FE fields
    χ = FEField(zeros(g.np), g, g1)
    f  = FEField(f,           g, g1)

    # solve 
    χ = solve_sf_test(χ, f, J, s, e)

    if showplots
        quickplot(χ, L"\chi", "images/chi.png")
        H(x) = sqrt(2 - x^2) - 1
        plot_profile(χ, 0.5, -H(0.5):1e-3:0, L"$\chi$ at $x = 0.5$",   L"z", "images/chi_profile.png")
        plot(x[ebot], χ.values[ebot], "o", ms=1)
        println(maximum(abs.(χ.values[ebot])))
        xlabel(L"x")
        ylabel(L"\chi(z = - H)")
        savefig("images/chi_bot.png")
        println("images/chi_bot.png")
        plt.close()
    end

    return χ
end

# χ = sf_test_res("gmsh", 3; showplots=true)
χ = sf_test_res("jc", 3; showplots=true)

println("Done.")