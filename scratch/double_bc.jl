using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SparseArrays
using SuiteSparse

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    -u'' = f
    u(0) = u₀
   u'(0) = 0
"""
function solve_double_bc(u, s, J, f, u₀)
    # indices
    N = u.g.np

    # setup matrix and vector
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)

    # stamp
    for k=1:u.g.nt
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # rhs
        b[u.g.t[k, :]] += M*f.values[u.g.t[k, :]]

        # add to global system
        for i=1:u.g.nn, j=1:u.g.nn
            push!(A, (u.g.t[k, i], u.g.t[k, j], K[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet b.c.
    A, b = add_dirichlet(A, b, u.g.e[end], u.g.e[1], u₀)
    # A, b = add_dirichlet(A, b, u.g.e[1], u₀)
    # A, b = add_dirichlet(A, b, u.g.e[end], 0)

    # solve
    sol = A\b

    # unpack
    u.values[:] = sol
    return u
end

function double_bc_res(; nref, order, showplots=false)
    # get grid
    np = 2^(nref + 2)
    h = 1.0/(np - 1)
    p = reshape(0:h:1, (np, 1))
    t = hcat(1:np-1, 2:np)
    e = [1, np]
    g = FEGrid(p, t, e, order)
    g1 = FEGrid(p, t, e, 1)

    # true res
    h = 1/(g.np - 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)

    # get Jacobians
    J = Jacobians(g1)

    # constructed solution
    x = g.p[:, 1]
    ua = @. -exp(x)*(x - 1)
    ua = FEField(ua, g, g1)
    u₀ = 1

    # rhs
    f = @. exp(x)*(1 + x)
    f = FEField(f, g, g1)

    # initialize FE field
    u = FEField(zeros(g.np), g, g1)

    # solve double b.c. problem
    u = solve_double_bc(u, s, J, f, u₀)

    if showplots
        fig, ax = subplots()
        ax.plot(g.p[:, 1], u.values, "o", ms=1, label="Numerical")
        ax.plot(g.p[:, 1], ua.values, "o", ms=0.5, label="Exact")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"u")
        ax.legend()
        savefig("images/u.png")
        println("images/u.png")
        plt.close()
    end

    err = L2norm(u - ua, s, J)
    return h, err
end

function double_bc_conv(; nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"$L^2$ Error")
    for o=1:3
        println("Order ", o)
        h = zeros(size(nrefs, 1))
        err = zeros(size(nrefs, 1))
        for i in eachindex(nrefs)
            println("\tRefinement ", nrefs[i])
            h[i], err[i] = double_bc_res(nref=nrefs[i], order=o)
        end
        ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^(o+1)], "k-", alpha=o/3, label=latexstring(L"$h^", o+1, L"$"))
        ax.loglog(h, err, "o", label="Order $o")
    end
    ax.legend(ncol=3, loc=(0.0, 1.05))
    savefig("images/double_bc.png")
    println("images/double_bc.png")
    plt.close()
end

h, err = double_bc_res(nref=0, order=3, showplots=true)

# double_bc_conv(nrefs=0:5)

println("Done.")