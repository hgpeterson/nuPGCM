using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SparseArrays
using SuiteSparse
using ProgressMeter

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    u = solve_laplace(V, f, u₀)

Solves -Δu = f with dirichlet b.c. u = u₀.
"""
function solve_laplace(u, s, J, f, u₀)
    println("np = $(u.g.np)")
    println("nt = $(u.g.nt)")

    # setup matrix and vector
    K = Tuple{Int64,Int64,Float64}[]
    b = zeros(u.g.np)

    # tag whether node of triangle is on boundary or not
    print("Creating edge tags... ")
    t₀ = time()
    edge_tags = [u.g.t[k, i] in u.g.e for k=1:u.g.nt, i in axes(u.g.t, 2)]
    println(time() - t₀, " s")

    # stamp
    @showprogress "Building... " for k=1:u.g.nt
        # calculate contribution to K from element k
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        Kᵏ = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]

        # calculate contribution to b from element k
        bᵏ = J.dets[k]*s.M*f[u.g.t[k, :]]

        # add to global system
        for i=1:u.g.nn, j=1:u.g.nn
            if edge_tags[k, i]
                # save for dirichlet
                continue
            end
            push!(K, (u.g.t[k, i], u.g.t[k, j], Kᵏ[i, j]))
        end
        b[u.g.t[k, :]] += bᵏ
    end

    # dirichlet b.c.
    for i in eachindex(u₀)
        push!(K, (u.g.e[i], u.g.e[i], 1))
        b[u.g.e[i]] = u₀[i]
    end

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), u.g.np, u.g.np)

    # solve
    print("Solving... ")
    t₀ = time()
    u.values[:] = K\b
    println(time() - t₀, " s")
    return u
end

function laplace_res(; nref, order, dim, showplots=false)
    # get grid
    if dim == 2
        gfile = "../meshes/gmsh/mesh$nref.h5"
    elseif dim == 3
        gfile = "../meshes/bowl3D/mesh$nref.h5"
    end
    g = FEGrid(gfile, order)
    g1 = FEGrid(gfile, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)

    # get Jacobians
    J = Jacobians(g1)

    # mesh resolution 
    h = 1/g.np^(1/dim)

    # solution
    x = g.p[:, 1]
    y = g.p[:, 2]
    if dim == 3
        z = g.p[:, 3]
    end
    if dim == 2
        ua = @. -exp(x^2)*sin(y)
    elseif dim == 3
        ua = @. -exp(x^2)*sin(y)*z^5
    end
    u₀ = ua[g.e]

    # pick f such that -∇²u = f
    if dim == 2
        f = @. exp(x^2)*(1 + 4*x^2)*sin(y)
    elseif dim == 3
        f = @. exp(x^2)*z^3*(20 + (1 + 4*x^2)*z^2)*sin(y)
    end

    # initialize FE field
    u = FEField(order, zeros(g.np), g, g1)
    ua = FEField(order, ua, g, g1)

    # solve laplace problem
    u = solve_laplace(u, s, J, f, u₀)

    if showplots
        if dim == 2
            quickplot(u, L"u", "images/u.png")
        elseif dim == 3
            write_vtk(g, "../output/laplace", ["u"=>u, "error"=>abs(u - ua)])
        end
    end

    # error
    err = L2norm(u - ua, s, J)
    return h, err
end

function laplace_convergence(; nrefs, dim)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u_a||_{L^2}$")
    orders = 1:2
    for o=orders
        println("Order ", o)
        h = zeros(size(nrefs, 1))
        err = zeros(size(nrefs, 1))
        for i in eachindex(nrefs)
            println("\tRefinement ", nrefs[i])
            h[i], err[i] = laplace_res(nref=nrefs[i], order=o, dim=dim)
        end
        ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^(o+1)], "k-", alpha=o/3, label=latexstring(L"$h^", o+1, L"$"))
        ax.loglog(h, err, "o", label="Order $o")
    end
    ax.legend(ncol=size(orders, 1), loc=(0.0, 1.05))
    savefig("images/laplace.png")
    println("images/laplace.png")
    plt.close()
end

laplace_res(nref=3, order=1, dim=3, showplots=true)
# laplace_convergence(nrefs=0:5, dim=2)