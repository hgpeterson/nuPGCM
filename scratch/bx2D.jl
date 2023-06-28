using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function get_M(g::Grid)
    J = g.J
    s = g.sfi
    M = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        Mᵏ = J.dets[k]*s.M 
        for i=1:g.nn, j=1:g.nn
            push!(M, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), g.np, g.np)
end

function get_C(g::Grid)
    J = g.J
    s = g.sfi
    C = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        Cᵏ = J.dets[k]*sum(s.C.*J.Js[k, :, 1], dims=1)[1, :, :]
        for i=1:g.nn, j=1:g.nn
            push!(C, (g.t[k, i], g.t[k, j], Cᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(C), (x->x[2]).(C), (x->x[3]).(C), g.np, g.np)
end

function get_bξ(g::Grid, b, f)
    M = get_M(g)
    C = get_C(g)
    M[1, :] .= 0
    M[1, 1] = 1
    M[g.np, :] .= 0
    M[g.np, g.np] = 1
    rhs = C*b
    rhs[1] = f(-1)
    rhs[g.np] = f(1)
    return M\rhs
end

function get_bx()
    # g = Grid(1, "meshes/valign2D/mesh2.h5")
    g = Grid(1, "meshes/gmsh/mesh4.h5")
    # H(x) = 1 - x^2
    # Hx(x) = -2x

    x = g.p[:, 1]
    z = g.p[:, 2]
    # δ = 0.1
    # b = @. z/2 + δ/2*exp(-(z + H(x))/δ) 
    # bx_a = @. -Hx(x)/2*exp(-(z + H(x))/δ)
    b = @. z^3
    bx_a = @. zeros(g.np)

    # etop = g.e["bdy"][abs.(g.p[g.e["bdy"], 2]) .< 1e-4]
    # ebot = g.e["bdy"][abs.(g.p[g.e["bdy"], 2]) .≥ 1e-4]
    # ebot = [etop[1]; ebot; etop[end]]
    # tplot(g.p, g.t)
    # plot(g.p[ebot, 1], g.p[ebot, 2], "o", ms=1)
    # axis("equal")
    # savefig("scratch/images/ebot.png")
    # println("scratch/images/ebot.png")
    # plt.close()

    # xbot = x[ebot]
    # perm = sortperm(xbot)
    # xbot = xbot[perm]
    # tbot = [i + j - 1 for i=1:size(ebot, 1)-1, j=1:2]
    # gbot = Grid(1, xbot, tbot, ebot)

    # # bbot = b[ebot][perm]
    # bbot = sin.(xbot .+ 1)
    # # bξ = get_bξ(gbot, bbot, x->-Hx(x)/2)
    # bξ = get_bξ(gbot, bbot, x->cos(x + 1))
    # err = FEField(abs.(bξ - cos.(xbot .+ 1)), g)
    # println(@sprintf("Max bξ Error: %1.1e at i=%d", maximum(err), argmax(err)))
    # fig, ax = plt.subplots(1)
    # ax.plot(xbot, bξ, label="Numerical")
    # ax.plot(xbot, -Hx(xbot)/2, "--", label=L"-H_x/2")
    # ax.legend()
    # ax.set_xlabel(L"x")
    # ax.set_ylabel(L"$\partial_\xi b = \partial_x b$ at $z = -H$")
    # savefig("scratch/images/bxi.png")
    # println("scratch/images/bxi.png")
    # plt.close()

    M = get_M(g)
    C = get_C(g)
    rhs = C*b
    # M[ebot, :] .= 0
    # for i ∈ eachindex(ebot)
    #     M[ebot[i], ebot[i]] = 1
    #     # rhs[ebot[i]] = bξ[i] # perm?
    #     rhs[ebot[i]] = -Hx(g.p[ebot[i], 1])/2
    #     # rhs[ebot[i]] = bx_a[ebot[i]]
    # end
    # M[etop, :] .= 0
    # for i ∈ etop
    #     M[i, i] = 1
    #     # rhs[i] = -Hx(g.p[i, 1])/2*exp(-H(g.p[i, 1])/δ)
    #     rhs[i] = bx_a[i]
    # end
    M[g.e["bdy"], :] .= 0
    for i ∈ g.e["bdy"]
        M[i, i] = 1
        rhs[i] = bx_a[i]
    end
    println(rank(M))
    println(g.np)
    bx = M\rhs

    err = FEField(abs.(bx - bx_a), g)
    println(@sprintf("Max Error: %1.1e at i=%d", maximum(err), argmax(err)))
    println(@sprintf("L2 Error: %1.1e", L2norm(err)))

    fig, ax, im = tplot(g.p, g.t, b, contour=true, cb_label=L"b")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/b2D.png")
    println("scratch/images/b2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, bx, contour=true, cb_label=L"b_x")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/bx2D.png")
    println("scratch/images/bx2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, bx_a, contour=true, cb_label=L"b_x^a")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/bxa2D.png")
    println("scratch/images/bxa2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, err.values, cb_label="Error")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/error2D.png")
    println("scratch/images/error2D.png")
    plt.close()

    return bx, bx_a
end

bx, bx_a = get_bx()

println("Done.")