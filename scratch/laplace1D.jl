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

function solve_laplace1D(u, s, J, f, u₀)
    # indices
    N = u.g.np
    println("N = $N")

    # setup matrix and vector
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)

    # stamp
    for k=1:u.g.nt
        # matrices
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
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
    A, b = add_dirichlet(A, b, u.g.e, u₀)

    # solve
    print("Solving... ")
    u.values[:] = A\b
    return u
end

function laplace1D_res(; nref, order, showplots=false)
    # get grid
    np = 2^nref + 1
    h = 1.0/(np - 1)
    p = reshape(0:h:1, (np, 1))
    t = hcat(1:np-1, 2:np)
    e = [1, np]
    g = FEGrid(p, t, e, order)
    g1 = FEGrid(p, t, e, 1)

    # get shape function integrals
    s = ShapeFunctionIntegrals(g.s, g.s)

    # get Jacobians
    J = Jacobians(g1)

    # rhs
    f = FEField(order, ones(g.np), g, g1)

    # dirichlet conditions
    u₀ = [0, 1]

    # true solution
    x = g.p[:, 1]
    ua = @. -1/2*x^2 + 3/2*x
    ua = FEField(order, ua, g, g1)

    # initialize FE field
    u = FEField(order, zeros(g.np), g, g1)

    # solve laplace problem
    u = solve_laplace1D(u, s, J, f, u₀)

    if showplots
        fig, ax = subplots()
        ax.plot(g.p[:, 1], u.values, "o", ms=1, label="Numerical")
        ax.plot(g.p[:, 1], ua.values, "o", ms=1, label="True")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"u")
        ax.legend()
        savefig("images/u.png")
        println("images/u.png")
        plt.close()
    end
end

laplace1D_res(nref=10, order=1, showplots=true)