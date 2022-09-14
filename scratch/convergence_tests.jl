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
    solves -Δu = f with dirichlet b.c. u = u₀
"""
function solve_poisson(p, t, e, C₀, f, u₀)
    # indices
    np = size(p, 1)
    nt = size(t, 1)

    # number of shape functions per triangle
    n = size(t, 2)

    # create global linear system using stamping method
    K = Tuple{Int64,Int64,Float64}[]
    b = zeros(np)
    for k = 1:nt
        # calculate contribution to K from element k
        Kᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = (shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η; dξ=1) + 
                              shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dη=1))
                Kᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to b from element k
        bᵏ = zeros(n)
        for i=1:n
            func(ξ, η) = f(ξ, η)*shape_func(C₀[k, i, :], ξ, η)
            bᵏ[i] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
        end

        # add to global system
        for i=1:n
            if t[k, i] in e
                # edge node, leave for dirichlet
                continue
            end
            for j=1:n
                push!(K, (t[k, i], t[k, j], Kᵏ[i, j]))
            end
            b[t[k, i]] += bᵏ[i]
        end
    end
    # dirichlet along edges
    for i in e
        push!(K, (i, i, 1))
    end
    b[e] = u₀

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), np, np)

    # solve
    return K\b
end

"""
    abs_err, h = poisson_res(res, shape_fns)
"""
function poisson_res(res, shape_fns)
    # geometry type
    # geo = "square"
    geo = "circle"

    # load mesh
    p, t, e = load_mesh("../meshes/$geo/mesh$res.h5")
    if shape_fns == "quad"
        p, t, e = add_midpoints(p, t)
    end
    ξ = p[:, 1]
    η = p[:, 2]

    # mesh resolution 
    h = 1/sqrt(size(p, 1))

    # get C₀
    C₀ = nuPGCM.get_shape_func_coeffs(p, t)

    # solution
    ua = @. -exp(ξ^2)*η

    # pick f such that -∇u = f
    f(ξ, η) = exp(ξ^2)*(2 + 4*ξ^2)*η

    # solve poisson problem
    u = solve_poisson(p, t, e, C₀, f, ua[e])

    # # plot
    # fig, ax, im = tplot(p, t, u)
    # cb = colorbar(im, ax=ax, label=L"u")
    # ax.axis("equal")
    # ax.set_xlabel(L"x")
    # ax.set_ylabel(L"y")
    # savefig("images/u.png")
    # println("images/u.png")
    # plt.close()

    # error
    abs_err = maximum(abs.(u - ua))
    return abs_err, h
end

"""
    poisson_convergence(rs)
"""
function poisson_convergence(rs)
    hs_l = zeros(size(rs, 1))
    hs_q = zeros(size(rs, 1))
    abs_err_l = zeros(size(rs, 1))
    abs_err_q = zeros(size(rs, 1))
    for i in eachindex(rs)
        println(rs[i])
        abs_err_l[i], hs_l[i] = poisson_res(rs[i], "linear")
        abs_err_q[i], hs_q[i] = poisson_res(rs[i], "quad")
    end

    fig, ax = subplots()
    ax.set_xlabel(L"Resolution $h$")
    ax.set_ylabel(L"Maximum absolute error $|u - u_a|$")
    ax.loglog([hs_l[1], hs_l[end]], [abs_err_l[1], abs_err_l[1]*(hs_l[end]/hs_l[1])^2], "k-",  label=L"$h^2$")
    ax.loglog([hs_q[1], hs_q[end]], [abs_err_q[1], abs_err_q[1]*(hs_q[end]/hs_q[1])^3], "k--", label=L"$h^3$")
    ax.loglog(hs_l, abs_err_l, "o", label="Linear")
    ax.loglog(hs_q, abs_err_q, "o", label="Quadratic")
    ax.legend(ncol=2)
    ax.set_xlim(0.9*hs_q[end], 1.1*hs_l[1])
    ax.set_ylim(0.5*abs_err_q[end], 2*abs_err_l[1])
    savefig("images/poisson.png")
    println("images/poisson.png")
    plt.close()

    println(@sprintf("Linear: %1.1f", log(abs_err_l[end-1]/abs_err_l[end])/log(hs_l[end-1]/hs_l[end])))
    println(@sprintf("Quad:   %1.1f", log(abs_err_q[end-1]/abs_err_q[end])/log(hs_q[end-1]/hs_q[end])))
end

# poisson_res(5, "linear")
poisson_convergence(0:5)

println("Done.")