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

function get_K(g::Grid)
    J = g.J
    s = g.sfi
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        Kᵏ = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        for i=1:g.nn, j=1:g.nn
            push!(K, (g.t[k, i], g.t[k, j], -Kᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np)
end

function evolve()
    ε² = 1e-2
    μ = 1
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    g = Grid(1, "meshes/valign2D/mesh2.h5")

    b = g.p[:, 2]

    # matrices
    M = get_M(g)
    K = get_K(g)
    LHS = lu(μ*ϱ*M - ε²*Δt/2*K)

    # solve
    n_steps = 10
    for i=1:n_steps
        RHS = μ*ϱ*M*b + Δt*ε²/2*K*b
        b = LHS\RHS
    end

    ba = [b_a(g.p[i, 2], n_steps*Δt, ε²/μ/ϱ, 1 - g.p[i, 1]^2) for i=1:g.np]
    println(@sprintf("Max Error: %1.1e", maximum(abs.(ba - b))))

    fig, ax, im = tplot(g.p, g.t, b, contour=true, vmax=1, cb_label=L"b")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/b2D.png")
    println("scratch/images/b2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, ba, contour=true, vmax=1, cb_label=L"b_a")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/ba2D.png")
    println("scratch/images/ba2D.png")
    plt.close()
    fig, ax, im = tplot(g.p, g.t, abs.(ba - b), cb_label="Error")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    savefig("scratch/images/error2D.png")
    println("scratch/images/error2D.png")
    plt.close()

    return b, ba
end

"""
Analytical solution to ∂t(b) = α ∂zz(b) with ∂z(b) = 0 at z = -H, 0
(truncated to Nth term in Fourier series).
"""
function b_a(z, t, α, H; N=10)
    if H == 0
        return 0
    end
    A(n) = 2*H*(1 + (-1)^(n+1))/(n^2*π^2)
    return -H/2 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
end

b, ba = evolve()
println("Done.")