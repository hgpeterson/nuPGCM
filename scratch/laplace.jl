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
    # create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]
    b = zeros(u.g.np)
    for k=1:u.g.nt
        # calculate contribution to K from element k
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        Kᵏ = J.dets[k]*dropdims(sum(s.K.*JJ, dims=(1, 2)), dims=(1, 2))

        # calculate contribution to b from element k
        bᵏ = J.dets[k]*s.M*f[u.g.t[k, :]]

        # add to global system
        for i=1:u.g.nn, j=1:u.g.nn
            push!(K, (u.g.t[k, i], u.g.t[k, j], Kᵏ[i, j]))
        end
        b[u.g.t[k, :]] += bᵏ
    end

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), u.g.np, u.g.np)

    # dirichlet along edges
    K, b = add_dirichlet(K, b, u.g.e, u₀)

    # remove zeros
    dropzeros!(K)

    # solve
    u.values[:] = K\b
    return u
end

"""
    h, err = laplace_res(nref, order)
"""
function laplace_res(nref, order; plot=false)
    # geometry type
    # geo = "square"
    geo = "circle"

    # get grid
    g = FEGrid("../meshes/$geo/mesh$nref.h5", order)
    g1 = FEGrid("../meshes/$geo/mesh$nref.h5", 1)
    x = g.p[:, 1]
    y = g.p[:, 2]

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)

    # get Jacobians
    J = Jacobians(g1)

    # mesh resolution 
    h = 1/sqrt(g.np)

    # solution
    ua = @. -exp(x^2)*y

    # pick f such that -∇u = f
    f = @. exp(x^2)*(2 + 4*x^2)*y

    # initialize FE field
    u = FEField(order, zeros(g.np), g, g1)
    ua = FEField(order, ua, g, g1)

    # solve laplace problem
    u = solve_laplace(u, s, J, f, ua.values[g.e])

    if plot
        quickplot(u, L"u", "images/u.png")
        quickplot(u - ua, "Error", "images/error.png")
    end

    # error
    err = L2norm(u - ua, s, J)
    return h, err
end

"""
    laplace_convergence(nrefs)
"""
function laplace_convergence(nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u_a||_{L^2}$")
    for o=1:3
        println("Order ", o)
        h = zeros(size(nrefs, 1))
        err = zeros(size(nrefs, 1))
        for i in eachindex(nrefs)
            println("\tRefinement ", nrefs[i])
            h[i], err[i] = laplace_res(nrefs[i], o)
        end
        ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^(o+1)], "k-", alpha=o/3, label=latexstring(L"$h^", o+1, L"$"))
        ax.loglog(h, err, "o", label="Order $o")
    end
    ax.legend(ncol=3, loc=(0.0, 1.05))
    # ax.set_xlim(0.5*h_q[end], 2*h_l[1])
    # ax.set_ylim(0.5*err_q[end], 2*err_l[1])
    savefig("images/laplace.png")
    println("images/laplace.png")
    plt.close()
end

laplace_res(5, 1; plot=true)
# laplace_convergence(0:3)