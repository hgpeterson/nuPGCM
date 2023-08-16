using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

# depth
H(x) = 1 - x[1]^2 - x[2]^2
# H(x) = 1 + 0*x[1]

function setup()
    ε² = 1e-2
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 1.
    β = 1.
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ(σ, H) = 1 + 0*σ*H
    ν(σ, H) = κ(σ, H)
    g_sfc1 = Grid(Triangle(order=1), "../meshes/circle/mesh2.h5")
    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, ν, κ, g_sfc1, nσ=0, chebyshev=false, advection=true)
    return m
end

function run(m)
    # b = FEField(x -> H(x)^3*(x[3]^2 + 2/3*x[3]^3), m.g2)
    b = FEField(x -> H(x)*x[3], m.g2)
    # b = FEField(x -> H(x)*x[3] + 0.1*exp(-H(x)*(x[3] + 1)/0.1), m.g2)
    # b = FEField(x -> exp(-(x[1]^2 + x[2]^2 + (H(x)*x[3] + 0.5)^2)/0.02), m.g2)

    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)
    # ωx = DGField(0, m.g1)
    # ωy = DGField(0, m.g1)
    # χx = DGField(0, m.g1)
    # χy = DGField(0, m.g1)
    # Ψ = FEField(0, m.g_sfc1)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, 0)

    t_final = 5e-2/(m.ε²/m.μ/m.ϱ)
    t_plot = t_final/5
    # t_final = 0.2
    # t_plot = t_final/10
    evolve!(m, s, t_final, t_plot)
    return s
end

# m = setup()
s = run(m)

function test_baroclinic()
    ε² = 1e-4
    μ = 1e0
    f = 1.
    x = 0.5
    y = 0
    H = 1 - x^2 - y^2
    nσ = 2^10
    σ = -1:1/(nσ-1):0
    z = σ*H
    ν = @. μ*(1e-2 + exp(-H*(σ + 1)/0.1))
    # ν = 1
    p = collect(σ)
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g = Grid(Line(order=1), p, t, e)
    A = nuPGCM.get_baroclinic_LHS(g, ν, H, ε², f)
    δ = 0.2
    b = @. H*σ + δ*exp(-H*(σ+1)/δ) - δ*exp(H*σ/δ)
    bz = @. 1 - exp(-H*(σ+1)/δ) - exp(H*σ/δ)
    bx = @. 2x*exp(-H*(σ+1)/δ)
    by = zeros(nσ)
    Ux = 0
    Uy = 1e-1
    τx = 0
    τy = 0
    r = nuPGCM.get_baroclinic_RHS(g, bx, by, Ux, Uy, τx, τy)
    sol = A\r
    ωx = sol[0*nσ+1:1*nσ]
    ωy = sol[1*nσ+1:2*nσ]
    χx = sol[2*nσ+1:3*nσ]
    χy = sol[3*nσ+1:4*nσ]
    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)
    ax[1, 1].plot(ωx, z)
    ax[1, 2].plot(ωy, z)
    ax[1, 3].plot(b, z)
    ax[2, 1].plot(χx, z)
    ax[2, 2].plot(χy, z)
    ax[2, 3].plot(bz, z)
    ax[1, 1].set_xlabel(L"\omega^x")
    ax[1, 2].set_xlabel(L"\omega^y")
    ax[1, 3].set_xlabel(L"b")
    ax[2, 1].set_xlabel(L"\chi^x")
    ax[2, 2].set_xlabel(L"\chi^y")
    ax[2, 3].set_xlabel(L"\partial_z b")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 1].set_ylim(-H, 0)
    ax[2, 1].set_ylim(-H, 0)
    savefig("images/test_baroclinic.png")
    println("images/test_baroclinic.png")
    plt.close()
end

# test_baroclinic()

println("Done.")