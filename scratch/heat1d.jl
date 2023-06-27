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

function b_a(z, t, α; N=10)
    # A(n) = 2*(1 + (-1)^(n+1))/(n^2*π^2)
    # return -1/2 + sum(A(n)*cos(n*π*z)*exp(-α*n^2*π^2*t) for n=1:2:N)
    A(n) = 8*(-1 + (-1)^n)/(n^4*π^4)
    return 1/6 + sum(A(n)*cos(n*π*z)*exp(-α*n^2*π^2*t) for n=1:2:N)
end

function solve_heat()
    nz = 2^5
    # z = -1:1/(nz-1):0 
    z = -(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2 
    p = reshape(z, (nz, 1))
    t = [i + j - 1 for i=1:nz-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nz])
    g = Grid(2, p, t, e)
    z = g.p[:, 1]

    ε² = 1e-2
    μ = 1
    ϱ = 1e-4
    T = 1e-2*μ*ϱ/ε²
    n_steps = 20
    Δt = T/n_steps

    # b = z
    b = @. z^2 + 2/3*z^3

    M = get_M(g)
    K = get_K(g)
    LHS = lu(μ*ϱ*M - ε²*Δt/2*K)
    RHS = μ*ϱ*M + ε²*Δt/2*K

    for i=1:n_steps
        b = LHS\(RHS*b)
    end

    ba = [b_a(z[i], T, ε²/μ/ϱ) for i ∈ eachindex(z)]
    err = FEField(abs.(b - ba), g)
    println(@sprintf("Max Error: %1.1e at i=%d", maximum(err), argmax(err)))
    println(@sprintf("L2 Error: %1.1e", L2norm(err)))

    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.plot(b, z, label="Numerical")
    ax.plot(ba, z, "--", label="Analytical")
    ax.legend()
    ax.set_xlabel(L"b")
    ax.set_ylabel(L"z")
    # ax.set_xlim(-1, 0)
    ax.set_xlim(0, 0.35)
    savefig("scratch/images/b.png")
    println("scratch/images/b.png")
    plt.close()
end

solve_heat()

# for O(Δt^2) convergence, need nz to be
# linear b, equal spaced z: 2^9
# linear b, chebyshev z: 2^8
# quad b, equal spaced z: 2^5
# quad b, chebyshev z: 2^5