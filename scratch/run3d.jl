using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    ε² = 1e-2
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 1.
    β = 1.
    H(x) = 1 - x[1]^2 - x[2]^2
    # τx(x) = -cos(π*x[2])
    τx(x) = 0.
    τy(x) = 0.

    g_sfc1 = Grid(1, "meshes/circle/mesh2.h5")

    # m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, g_sfc1)

    # Ψ = m.barotropic_LHS\m.barotropic_RHS_τ
    # Ψ = FEField(Ψ, m.g_sfc1)
    # nuPGCM.quick_plot(Ψ, L"\Psi", "$out_folder/psi.png")

    δ = 0.1
    # b = FEField(x -> H(x)*x[3], m.g2)
    # b = FEField(x -> H(x)*x[3] + δ*exp(-H(x)*(x[3] + 1)/δ), m.g2)
    b = FEField(x -> δ*exp(-H(x)*(x[3] + 1)/δ), m.g2)

    H = m.H
    σ = m.σ
    nσ = m.nσ
    Dxs = m.Dxs
    Dys = m.Dys
    k = 100
    i = 1
    ig = g_sfc1.t[k, i]
    display(g_sfc1.p[g_sfc1.t[k, :], :])
    display(H[ig])
    display(m.Hx[k, i])
    display(m.Hy[k, i])
    bx = Dxs[k, i]*b.values
    by = Dys[k, i]*b.values
    z_dg = zeros(2nσ-2)
    for j=1:nσ-1
        z_dg[2j-1] = H[ig]*σ[j]
        z_dg[2j] = H[ig]*σ[j+1]
    end
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
    bx_a = @. -m.Hx[k, i]*exp(-H[ig]*(σ + 1)/δ)
    by_a = @. -m.Hy[k, i]*exp(-H[ig]*(σ + 1)/δ)
    ax[1].plot(bx_a, H[ig]*σ)
    ax[1].plot(bx, z_dg, "--")
    ax[1].set_xlabel(L"\partial_x b")
    ax[1].set_ylabel(L"z")
    ax[2].plot(by_a, H[ig]*σ)
    ax[2].plot(by, z_dg, "--")
    ax[2].set_xlabel(L"\partial_y b")
    savefig("scratch/images/bxby.png")
    println("scratch/images/bxby.png")
    plt.close()

    # b = FEField(x->x[1]*x[2], m.g2)
    # nuPGCM.quick_plot(FEField(x->x[2]^2-x[1]^2, g_sfc1), L"J(1/H, \gamma)", "$out_folder/JEBAR_a.png")
    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, 0)

    return m, s

    # m = ModelSetup3D()
    # # b = [FEField(x -> x[3], g) for g ∈ m.b_cols]
    # b = [FEField(x -> H(x)*x[3]^2 + 2/3*x[3]^3, g) for g ∈ m.b_cols]
    # # δ = 0.1
    # # b = [FEField(x -> x[3] + δ*exp(-(x[3] + H(x))/δ), g) for g ∈ m.b_cols]
    # ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    # s = ModelState3D(b, ωx, ωy, χx, χy, 0)
    # # evolve!(m, s)
    # return m, s
end

# m = bowl()
m, s = bowl()

println("Done.")