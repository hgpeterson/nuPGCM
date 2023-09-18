using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

# depth
H(x) = 1 - x[1]^2 - x[2]^2

function setup()
    μ = 1e0
    # ε² = 4e-6
    ε² = 1e-2
    # ϱ = 7e-4
    ϱ = 1e0
    # ϱ = 1e-4
    # Δt = 1e-3*μ*ϱ/ε²
    Δt = 1e-4*μ*ϱ/ε²
    f = 1.
    β = 0.
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ(σ, H) = 1 + 0*σ*H
    ν(σ, H) = κ(σ, H)
    g_sfc1 = Grid(Triangle(order=1), "../meshes/circle/mesh2.h5")
    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, ν, κ, g_sfc1, chebyshev=false, advection=true)
    save_setup(m)
    return m
end

function run(m)
    # b = FEField(x -> H(x)*x[3], m.g2)
    b = FEField(x -> H(x)*x[3] + 0.1*exp(-H(x)*(x[3] + 1)/0.1), m.g2)

    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, 0)
    # nuPGCM.plot_u(m, s, 0)
    # s.b.values[:] = FEField(x -> exp(-((x[1] - 0.5)^2 + x[2]^2 + (H(x)*x[3] + 0.75)^2)/0.02), m.g2).values
    s.b.values[:] = FEField(x -> exp(-((x[1] - 0.8)^2 + x[2]^2 + (H(x)*x[3] + H([0, 0.8]))^2)/0.02), m.g2).values

    t_final = 500*m.Δt
    t_plot = t_final/50
    evolve!(m, s, t_final, t_plot)
    return s
end

# m = setup()
m = load_setup_3D("$out_folder/setup.h5")
s = run(m)
# s = load_state_3D("$out_folder/state.h5")

function compare_profiles(m, s, m2D, s2D, x, y)
    k_sfc = nuPGCM.get_k([x, y], m.g_sfc1, m.g_sfc1.el)
    ξ_sfc = nuPGCM.transform_to_ref_el(m.g_sfc1.el, [x, y], m.g_sfc1.p[m.g_sfc1.t[k_sfc, :], :])

    σ = m.σ
    nσ = m.nσ
    H = m.H(ξ_sfc, k_sfc)
    z = σ*H
    k_ws = nuPGCM.get_k_ws(k_sfc, nσ)
    k_ws = [k_ws; k_ws[end]]
    ξ_ws = [nuPGCM.transform_to_ref_el(m.g1.el, [x, y, σ[i]], m.g1.p[m.g1.t[k_ws[i], :], :]) for i=1:nσ]

    ωx_fe = FEField(s.ωx)
    ωy_fe = FEField(s.ωy)
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    ωxs = [ωx_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    ωys = [ωy_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    χxs = [χx_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    χys = [χy_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    bs = [s.b(ξ_ws[i], k_ws[i]) for i=1:nσ]
    bzs = differentiate(bs, z)

    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)

    ax[1, 1].set_xlabel(L"\omega^x")
    ax[1, 2].set_xlabel(L"\omega^y")
    ax[1, 3].set_xlabel(L"b")
    ax[2, 1].set_xlabel(L"\chi^x")
    ax[2, 2].set_xlabel(L"\chi^y")
    ax[2, 3].set_xlabel(L"\partial_z b")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 2].set_title(latexstring(@sprintf("\$x = %1.1f \\quad y = %1.1f\$", x, y)))
    ax[1, 1].set_ylim(-H, 0)
    ax[2, 1].set_ylim(-H, 0)
    for a ∈ ax
        a.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    end

    ax[1, 1].plot(ωxs, z, label="3D")
    ax[1, 2].plot(ωys, z)
    ax[1, 3].plot(bs,  z)
    ax[2, 1].plot(χxs, z)
    ax[2, 2].plot(χys, z)
    ax[2, 3].plot(bzs, z)
    ix = argmin(abs.(m2D.ξ .- 0.5))
    H = m2D.H[ix]
    z = m2D.z[ix, :]
    ωx = -1/H*differentiate(s2D.uη[ix, :], m2D.σ)
    ωy =  1/H*differentiate(s2D.uξ[ix, :], m2D.σ)
    χx =  H*cumtrapz(s2D.uη[ix, :], m2D.σ)
    χy = -H*cumtrapz(s2D.uξ[ix, :], m2D.σ)
    b = s2D.b[ix, :]
    bz = 1/H*differentiate(s2D.b[ix, :], m2D.σ)
    ax[1, 1].plot(ωx, z, "k--", lw=0.5, label="2D")
    ax[1, 2].plot(ωy, z, "k--", lw=0.5)
    ax[1, 3].plot(b,  z, "k--", lw=0.5)
    ax[2, 1].plot(χx, z, "k--", lw=0.5)
    ax[2, 2].plot(χy, z, "k--", lw=0.5)
    ax[2, 3].plot(bz, z, "k--", lw=0.5)
    ax[1, 1].legend()
    savefig("$out_folder/profiles2Dvs3D.png")
    println("$out_folder/profiles2Dvs3D.png")
    plt.close()
end

# compare_profiles(m, s, m2D, s2D, 0.5, 0)

println("Done.")