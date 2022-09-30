using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
    u = solve_laplace_z(g, s, J, δ, f, u₀)

Solve linear system representing the problem
    -δ² u_zz + u = f  on Ω
               u = u₀ on ∂Ω
in a finite element basis.
"""
function solve_laplace_z(g, s, J, δ, f, u₀)
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt        
        # calculate contribution to K from element k
        Kᵏ = δ^2*abs(J.J[k])*(s.φξφξ*J.ξy[k]^2 + 
                              s.φξφη*J.ξy[k]*J.ηy[k] +
                              s.φηφξ*J.ηy[k]*J.ξy[k] +
                              s.φηφη*J.ηy[k]^2)

        # calculate contribution to M from element k
        Mᵏ = abs(J.J[k])*s.φφ

        # calculate contribution to b from element k
        bᵏ = abs(J.J[k])*s.φφ*f[g.t[k, :]]

        # add to global system
        for i=1:g.nn
            for j=1:g.nn
                push!(A, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
            end
            b[g.t[k, i]] += bᵏ[i]
        end
    end

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), g.np, g.np)

    # dirichlet along edges
    A[g.e, :] .= 0
    A[diagind(A)[g.e]] .= 1
    b[g.e] = u₀

    return A\b
end

"""
    h, err = laplace_z_res(nref, order)
"""
function laplace_z_res(nref, order; plot=false)
    # parameter
    δ = 0.05

    # geometry type
    geo = "jc"

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

    # analytical solution
    ua = @. 100*sin(x)*(y + 0.25)^2

    # forcing 
    f = @. 100*sin(x)*(y + 0.25)^2 - 200*δ^2*sin(x)

    # dirichlet boundary
    u₀ = ua[g.e]

    # solve laplace_z problem
    u = solve_laplace_z(g, s, J, δ, f, u₀)

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
    laplace_z_convergence(nrefs)
"""
function laplace_z_convergence(nrefs)
    h_l = zeros(size(nrefs, 1))
    h_q = zeros(size(nrefs, 1))
    err_l = zeros(size(nrefs, 1))
    err_q = zeros(size(nrefs, 1))
    for i in eachindex(nrefs)
        println(nrefs[i])
        h_l[i], err_l[i] = laplace_z_res(nrefs[i], 1)
        h_q[i], err_q[i] = laplace_z_res(nrefs[i], 2)
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
    savefig("images/laplace_z.png")
    println("images/laplace_z.png")
    plt.close()
end

laplace_z_convergence(0:5)
# laplace_z_res(3, 2; plot=true)