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
        Kᵏ = abs(J.J[k])*(s.φξφξ*(J.ξx[k]^2       + J.ξy[k]^2) + 
                          s.φξφη*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                          s.φηφξ*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                          s.φηφη*(J.ηx[k]^2       + J.ηy[k]^2))

        # calculate contribution to b from element k
        bᵏ = abs(J.J[k])*s.φφ*f[u.g.t[k, :]]

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

    # solve laplace problem
    u = solve_laplace(u, s, J, f, ua[g.e])

    if plot
        quickplot(u, L"u", "images/u.png")
    end

    # error
    err = L2norm(g, s, J, u.values - ua)
    return h, err
end

"""
    laplace_convergence(nrefs)
"""
function laplace_convergence(nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u^a||_{L^2}$")
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

# laplace_res(0, 3; plot=true)
laplace_convergence(0:3)