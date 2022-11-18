using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
    u = solve_laplace_hydro(g, s, J, δ, f, u₀)

Solve linear system representing the problem
    -δ² u_zz + u = f  on Ω
               u = u₀ on ∂Ω
in a finite element basis.
"""
function solve_laplace_hydro(g, s, J, δ, f, u₀)
    A = Tuple{Int64,Int64,Float64}[]
    b = zeros(g.np)
    for k=1:g.nt        
        # calculate contribution to K from element k
        JJ = J.Js[k, :, 2]*J.Js[k, :, 2]'
        Kᵏ = δ^2*J.dets[k]*dropdims(sum(s.K.*JJ, dims=(1, 2)), dims=(1, 2))

        # calculate contribution to M from element k
        Mᵏ = J.dets[k]*s.M

        # calculate contribution to b from element k
        bᵏ = Mᵏ*f[g.t[k, :]]

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
    A, b = add_dirichlet(A, b, g.e, u₀)

    return A\b
end

"""
    h, err = laplace_hydro_res(nref, order)
"""
function laplace_hydro_res(nref, order; plot=false)
    # parameter
    δ = 0.05

    # geometry type
    geo = "jc"

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

    # analytical solution
    ua = @. 100*sin(x)*(y + 0.25)^2

    # forcing 
    f = @. 100*sin(x)*(y + 0.25)^2 - 200*δ^2*sin(x)

    # dirichlet boundary
    u₀ = ua[g.e]

    # solve laplace_hydro problem
    u = solve_laplace_hydro(g, s, J, δ, f, u₀)

    u = FEField(order, u, g, g1)
    ua = FEField(order, ua, g, g1)
    abs_err = abs(u - ua)

    if plot
        quickplot(u, L"u", "images/u.png")
        quickplot(ua, L"u_a", "images/ua.png")
        quickplot(abs_err, L"|u - u_a|", "images/e.png")
    end

    # error
    err = L2norm(abs_err, s, J)
    return h, err
end

"""
    laplace_hydro_convergence(nrefs)
"""
function laplace_hydro_convergence(nrefs)
    h_l = zeros(size(nrefs, 1))
    h_q = zeros(size(nrefs, 1))
    err_l = zeros(size(nrefs, 1))
    err_q = zeros(size(nrefs, 1))
    for i in eachindex(nrefs)
        println(nrefs[i])
        h_l[i], err_l[i] = laplace_hydro_res(nrefs[i], 1)
        h_q[i], err_q[i] = laplace_hydro_res(nrefs[i], 2)
    end

    fig, ax = subplots(1)
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Error $||u - u_a||_{L^2}$")
    ax.loglog([h_l[1], h_l[end]], [err_l[1], err_l[1]*(h_l[end]/h_l[1])^2], "k-",  label=L"$h^2$")
    ax.loglog([h_q[1], h_q[end]], [err_q[1], err_q[1]*(h_q[end]/h_q[1])^3], "k--", label=L"$h^3$")
    ax.loglog(h_l, err_l, "o", label="Linear")
    ax.loglog(h_q, err_q, "o", label="Quadratic")
    ax.legend(ncol=2)
    ax.set_xlim(0.5*h_q[end], 2*h_l[1])
    ax.set_ylim(0.5*err_q[end], 2*err_l[1])
    savefig("images/laplace_hydro.png")
    println("images/laplace_hydro.png")
    plt.close()
end

# laplace_hydro_convergence(0:5)
laplace_hydro_res(5, 1; plot=true)