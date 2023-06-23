using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays

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
            push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np)
end

function b_a(z, t, α; N=10)
    A(n) = 2*(1 + (-1)^(n+1))/(n^2*π^2)
    return -1/2 + sum(A(n)*cos(n*π*z)*exp(-α*n^2*π^2*t) for n=1:2:N)
end

function solve_heat()
    z = -1:0.01:0
    nz = size(z, 1)
    p = reshape(z, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nz])
    g = Grid(1, p, t, e)

    ε² = 1e-2
    μ = 1
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²

    b = z

    M = get_M(g)
    K = get_K(g)
    LHS = lu(μ*ϱ*M + ε²*Δt/2*K)

    n_steps = 10
    for i=1:n_steps
        RHS = μ*ϱ*M*b - Δt*ε²/2*K*b
        b = LHS\RHS
    end

    t = Δt*n_steps
    bt(z) = b_a(z, t, ε²/μ/ϱ)

    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.plot(b, z, label="Numerical")
    ax.plot(bt.(z), z, "--", label="Analytical")
    ax.legend()
    ax.set_xlabel(L"b")
    ax.set_ylabel(L"z")
    ax.set_xlim(-1, 0)
    savefig("scratch/images/b.png")
    println("scratch/images/b.png")
    plt.close()
end

solve_heat()