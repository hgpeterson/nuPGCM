using nuPGCM
using PyPlot
using LinearAlgebra
using SparseArrays
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function build_K(g, κ)
    κ = FEField(κ, g)
    J = g.J
    el = g.el
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt
        # ∫ ν ∂φᵢ∂φⱼ
        σ(ξ) = transform_from_ref_el(el, ξ, g.p[g.t[k, :]])
        κK = [nuPGCM.ref_el_quad(ξ -> κ(σ(ξ), k)*φξ(el, ξ, i)*φξ(el, ξ, j)*J.Js[k, 1, 1]^2*J.dets[k], el) for i=1:el.n, j=1:el.n]
        for i=1:el.n, j=1:el.n
            push!(K, (g.t[k, i], g.t[k, j], κK[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np))
end

function solve_heat()
    # grid
    nσ = 64
    # σ = -1:1/(nσ-1):0 
    σ = -(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2 
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g = Grid(Line(order=1), σ, t, e)

    # params
    H = 1.0
    κ = @. 1e-2 + exp(-H*(σ + 1)/0.1)
    α = 1e1
    Δt = 1e-5
    n_steps = 100

    # b₀
    b = H*σ
    ∫b₀ = trapz(b, σ*H)
    println("∫b₀ = ", ∫b₀)

    M = nuPGCM.mass_matrix(g)
    K = build_K(g, κ)
    LHS = M + α/H^2*Δt/2*K
    LHS[nσ, :] .= 0
    LHS[nσ, nσ] = 1
    LHS = lu(LHS)
    RHS = M - α/H^2*Δt/2*K
    RHS[nσ, :] .= 0

    for i=1:n_steps
        b = LHS\(RHS*b)
    end

    # integral conservation
    ∫b = trapz(b, σ*H)
    println("∫b = ", ∫b)
    Δb = abs(∫b - ∫b₀)
    println("Δb = ", Δb)
    Δb_pct = 100*Δb/abs(∫b₀)
    println("Δb_pct = ", Δb_pct)

    # plot
    σ_hr = -1:0.001:0
    bz = [∂(FEField(b, g), σ_hr[i], 1) for i ∈ eachindex(σ_hr)]
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
    ax[1].plot(b, σ*H)
    ax[1].set_xlim(-H, 0)
    ax[2].plot(bz, σ_hr*H)
    ax[1].set_xlabel(L"b")
    ax[2].set_xlabel(L"\partial_z b")
    ax[1].set_ylabel(L"z")
    savefig("images/b.png")
    println("images/b.png")
    plt.close()
end

solve_heat()