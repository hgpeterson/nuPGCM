using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SparseArrays
using SuiteSparse
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    u = solve_poisson(V, f, u₀)

Solves -Δu = f with dirichlet b.c. u = u₀.
"""
function solve_poisson(g, s, J, f, u₀)
    # create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt
        # calculate contribution to K from element k
        Kᵏ = abs(J.J[k])*(s.φξφξ*(J.ξx[k]^2       + J.ξy[k]^2) + 
                          s.φξφη*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                          s.φηφξ*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                          s.φηφη*(J.ηx[k]^2       + J.ηy[k]^2))

        # calculate contribution to b from element k
        bᵏ = abs(J.J[k])*s.φφ*f[g.t[k, :]]

        # add to global system
        for i=1:g.nn
            for j=1:g.nn
                push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
            end
            b[g.t[k, i]] += bᵏ[i]
        end
    end

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), g.np, g.np)

    # dirichlet along edges
    K[g.e, :] .= 0
    K[diagind(K)[g.e]] .= 1
    b[g.e] = u₀

    # solve
    return K\b
end

"""
    h, err = poisson_res(nref, order)
"""
function poisson_res(nref, order; plot=false)
    # geometry type
    # geo = "square"
    geo = "circle"

    # get grid
    g = Grid("../meshes/$geo/mesh$nref.h5", order)
    x = g.p[:, 1]
    y = g.p[:, 2]

    # get shape functions
    sf = ShapeFunctions(order)

    # get shape function integrals
    s = ShapeFunctionIntegrals(sf, sf)

    # get Jacobians
    J = Jacobians(g)

    # mesh resolution 
    h = 1/sqrt(g.np)

    # solution
    ua = @. -exp(x^2)*y

    # pick f such that -∇u = f
    f = @. exp(x^2)*(2 + 4*x^2)*y

    # solve poisson problem
    u = solve_poisson(g, s, J, f, ua[g.e])

    if plot
        quickplot(g, u, L"u", "images/u.png")
        quickplot(g, ua, L"u^a", "images/ua.png")
        quickplot(g, abs.(u - ua), L"|u - u^a|", "images/e.png")
    end

    # error
    err = L2norm(g, s, J, u - ua)
    return h, err
end

"""
    quickplot(g, u, clabel, ofile)
"""
function quickplot(g, u, clabel, ofile)
    fig, ax, im = tplot(g.p, g.t, u)
    cb = colorbar(im, ax=ax, label=clabel)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    savefig(ofile)
    println(ofile)
    plt.close()
end

"""
    poisson_convergence(nrefs)
"""
function poisson_convergence(nrefs)
    h_l = zeros(size(nrefs, 1))
    h_q = zeros(size(nrefs, 1))
    err_l = zeros(size(nrefs, 1))
    err_q = zeros(size(nrefs, 1))
    for i in eachindex(nrefs)
        println(nrefs[i])
        h_l[i], err_l[i] = poisson_res(nrefs[i], 1)
        h_q[i], err_q[i] = poisson_res(nrefs[i], 2)
    end

    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u^a||_{L^2}$")
    ax.loglog([h_l[1], h_l[end]], [err_l[1], err_l[1]*(h_l[end]/h_l[1])^2], "k-",  label=L"$h^2$")
    ax.loglog([h_q[1], h_q[end]], [err_q[1], err_q[1]*(h_q[end]/h_q[1])^3], "k--", label=L"$h^3$")
    ax.loglog(h_l, err_l, "o", label="Linear")
    ax.loglog(h_q, err_q, "o", label="Quadratic")
    ax.legend(ncol=2)
    ax.set_xlim(0.5*h_q[end], 2*h_l[1])
    ax.set_ylim(0.5*err_q[end], 2*err_l[1])
    savefig("images/poisson.png")
    println("images/poisson.png")
    plt.close()
end

# poisson_res(3, 2; plot=true)
poisson_convergence(0:5)

println("Done.")