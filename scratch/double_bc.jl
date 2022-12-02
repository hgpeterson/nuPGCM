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
        v' = f
    u' - v = 0
"""
function solve_double_bc(u, v, s, J, f, u₀, v₀)
    # indices
    umap = 1:u.g.np
    vmap = umap[end] .+ (1:v.g.np)
    N = vmap[end]
    println("N = $N")

    # setup matrix and vector
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(N)

    # stamp
    for k=1:u.g.nt
        # matrices
        C_vu = J.dets[k]*sum(s.vu.C.*J.Js[k, :, 1], dims=1)[1, :, :]
        C_uv = J.dets[k]*sum(s.uv.C.*J.Js[k, :, 1], dims=1)[1, :, :]
        M_uu = J.dets[k]*s.uu.M
        M_vv = J.dets[k]*s.vv.M

        # rhs
        b[umap[u.g.t[k, :]]] = M_uu*f.values[u.g.t[k, :]]

        # add to global system
        for i=1:u.g.nn, j=1:v.g.nn
            push!(A, (umap[u.g.t[k, i]], vmap[v.g.t[k, j]], C_vu[i, j]))
        end
        for i=1:v.g.nn, j=1:u.g.nn
            push!(A, (vmap[v.g.t[k, i]], umap[u.g.t[k, j]], C_uv[i, j]))
        end
        for i=1:v.g.nn, j=1:v.g.nn
            push!(A, (vmap[v.g.t[k, i]], vmap[v.g.t[k, j]], M_vv[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet b.c.
    A, b = add_dirichlet(A, b, umap[u.g.e[1]], u₀)
    A, b = add_dirichlet(A, b, vmap[v.g.e[1]], v₀)

    # solve
    print("Solving... ")
    sol = A\b

    # unpack
    u.values[:] = sol[umap]
    v.values[:] = sol[vmap]
    return u, v
end

function double_bc_res(; nref, order_u, order_v, showplots=false)
    # get grid
    np = 2^nref + 1
    h = 1.0/(np - 1)
    p = reshape(0:h:1, (np, 1))
    t = hcat(1:np-1, 2:np)
    e = [1, np]
    gu = FEGrid(p, t, e, order_u)
    gv = FEGrid(p, t, e, order_v)
    g1 = FEGrid(p, t, e, 1)

    # get shape function integrals
    uu = ShapeFunctionIntegrals(gu.s, gu.s)
    uv = ShapeFunctionIntegrals(gu.s, gv.s)
    vu = ShapeFunctionIntegrals(gv.s, gu.s)
    vv = ShapeFunctionIntegrals(gv.s, gv.s)
    s = (uu = uu, uv = uv, vu = vu, vv = vv)

    # get Jacobians
    J = Jacobians(g1)

    # rhs
    f = FEField(order_u, ones(gu.np), gu, g1)

    # dirichlet conditions
    u₀ = 0
    v₀ = 0

    # initialize FE field
    u = FEField(order_u, zeros(gu.np), gu, g1)
    v = FEField(order_v, zeros(gv.np), gv, g1)

    # solve laplace problem
    u, v = solve_double_bc(u, v, s, J, f, u₀, v₀)

    if showplots
        fig, ax = subplots()
        ax.plot(gu.p[:, 1], u.values, "o", ms=1, label=L"u")
        ax.plot(gv.p[:, 1], v.values, "o", ms=1, label=L"v")
        ax.set_xlabel(L"x")
        ax.legend()
        ax.set_xlim(0, 0.01)
        ax.set_ylim(0, 0.01)
        savefig("images/uv.png")
        println("images/uv.png")
        plt.close()
    end
end

double_bc_res(nref=10, order_u=1, order_v=1, showplots=true)