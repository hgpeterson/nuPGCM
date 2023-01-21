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
    u, v = solve_pg_vort_1D(u, v, f, diri, J, s)

       -u'' = f₁, 
    v'' + u = f₂
    v(0) = v(1) = u(1) = 0
    v'(0) = 0
"""
function solve_pg_vort_1D(u, v, f, diri, J, s)
    # unpack grids
    g1 = u.g1
    g = u.g
    # indices
    umap = 1:g.np
    vmap = (g.np+1):2*g.np
    N = vmap[end]

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:g1.nt
        # stiffness and mass matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # RHSs
        r[umap[g.t[k, :]]] += M*f.f1.values[g.t[k, :]]
        r[vmap[g.t[k, :]]] += M*f.f2.values[g.t[k, :]]

        for i=1:g.nn, j=1:g.nn
            # indices
            ui = umap[g.t[k, :]]
            vi = vmap[g.t[k, :]]

            # -u''
            push!(A, (ui[i], ui[j], K[i, j]))

            # +v''
            push!(A, (vi[i], vi[j], -K[i, j]))
            # +u
            push!(A, (vi[i], ui[j], M[i, j]))
        end
    end

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # dirichlet
    A, r = add_dirichlet(A, r, umap[g.e[1]], vmap[g.e[1]], diri.v0) 
    A, r = add_dirichlet(A, r, umap[g.e[end]], diri.u1) 
    A, r = add_dirichlet(A, r, vmap[g.e[end]], diri.v1)

    # remove zeros
    dropzeros!(A)

    # solve
    sol = A\r

    # reshape to get ω and χ
    u.values[:] = sol[umap]
    v.values[:] = sol[vmap]
    return u, v
end

function pg_vort_1D_res(; nref, order, showplots=false)
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
    ua = @. sin(x)
    va = @. -exp(x)*(x - 1)
    diri = (u0=ua[g.e[1]], u1=ua[g.e[end]],
            v0=va[g.e[1]], v1=va[g.e[end]]
           )
    ua = FEField(ua, g, g1)
    va = FEField(va, g, g1)

    # forcing
    f1 = @. sin(x)
    f2 = @. -exp(x)*(x + 1) + sin(x)
    f1 = FEField(f1, g, g1)
    f2 = FEField(f2, g, g1)
    f = (f1=f1, f2=f2)

    # initialize FE fields
    u = FEField(zeros(g.np), g, g1)
    v = FEField(zeros(g.np), g, g1)    

    # solve 
    u, v = solve_pg_vort_1D(u, v, f, diri, J, s)

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
        fig, ax = subplots()
        ax.plot(g.p[:, 1], v.values, "o", ms=1, label="Numerical")
        ax.plot(g.p[:, 1], va.values, "o", ms=0.5, label="Exact")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"v")
        ax.legend()
        savefig("images/v.png")
        println("images/v.png")
        plt.close()
    end

    err = L2norm(u - ua, s, J) + L2norm(v - va, s, J)

    return h, err
end

function pg_vort_1D_conv(; nrefs)
    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"$L^2$ Error")
    for o=1:3
        println("Order ", o)
        h = zeros(size(nrefs, 1))
        err = zeros(size(nrefs, 1))
        for i in eachindex(nrefs)
            println("\tRefinement ", nrefs[i])
            h[i], err[i] = pg_vort_1D_res(nref=nrefs[i], order=o)
        end
        ax.loglog([h[1], h[end]], [err[1], err[1]*(h[end]/h[1])^(o+1)], "k-", alpha=o/3, label=latexstring(L"$h^", o+1, L"$"))
        ax.loglog(h, err, "o", label="Order $o")
    end
    ax.legend(ncol=3, loc=(0.0, 1.05))
    savefig("images/pg_vort_1D.png")
    println("images/pg_vort_1D.png")
    plt.close()
end

h, err = pg_vort_1D_res(nref=3, order=3, showplots=true)
# pg_vort_1D_conv(nrefs=0:3)

println("Done.")