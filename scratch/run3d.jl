using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

# depth
H(x) = 1 - x[1]^2 - x[2]^2
# H(x) = 1 + 0*x[1]

function setup()
    ε² = 4e-6
    μ = 1e0
    ϱ = 7e-4
    Δt = 1e-3*μ*ϱ/ε²
    println(√(2*ε²))
    f = 1.
    β = 0.
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ(σ, H) = 1 + 0*σ*H
    ν(σ, H) = κ(σ, H)
    g_sfc1 = Grid(Triangle(order=1), "../meshes/circle/mesh4.h5")
    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, ν, κ, g_sfc1, nσ=0, chebyshev=true, advection=false)
    return m
end

function run(m)
    # b = FEField(x -> H(x)^3*(x[3]^2 + 2/3*x[3]^3), m.g2)
    b = FEField(x -> H(x)*x[3], m.g2)
    # b = FEField(x -> H(x)*x[3] + 0.1*exp(-H(x)*(x[3] + 1)/0.1), m.g2)
    # b = FEField(x -> exp(-(x[1]^2 + x[2]^2 + (H(x)*x[3] + 0.5)^2)/0.02), m.g2)

    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, 0)
    # s.b.values[:] = FEField(x -> exp(-((x[1] - 0.5)^2 + x[2]^2 + (H(x)*x[3] + 0.75)^2)/0.02), m.g2).values

    t_final = 5e-2/(m.ε²/m.μ/m.ϱ)
    t_plot = t_final
    # t_final = 40
    # t_plot = t_final/40
    evolve!(m, s, t_final, t_plot)
    return s
end

# m = setup()
# s = run(m)

function compare_profiles(m, s, m2D, s2D, x, y)
    k_sfc = nuPGCM.get_k([x, y], m.g_sfc1, m.g_sfc1.el)

    σ = m.σ
    nσ = m.nσ
    H = m.H([x, y])
    z = σ*H
    k_ws = nuPGCM.get_k_ws(k_sfc, nσ)
    k_ws = [k_ws; k_ws[end]]

    ωx_fe = FEField(s.ωx)
    ωy_fe = FEField(s.ωy)
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    ωxs = [ωx_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    ωys = [ωy_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    χxs = [χx_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    χys = [χy_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    bs = [s.b([x, y, σ[i]], k_ws[i]) for i=1:nσ]
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
    ax[2, 1].plot(χys, z)
    ax[2, 2].plot(-χxs, z)
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

using PyCall
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
pc = 1/6
function compare_ux_uy_b(m, s, m2D, s2D, x, y)
    ix = argmin(abs.(m2D.ξ .- 0.5))
    y = m2D.ξ[ix]
    H = m2D.H[ix]
    k_sfc = nuPGCM.get_k([x, y], m.g_sfc1, m.g_sfc1.el)

    σ = m.σ
    nσ = m.nσ
    z = σ*H
    k_ws = nuPGCM.get_k_ws(k_sfc, nσ)
    k_ws = [k_ws; k_ws[end]]

    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    χxs = [χx_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    χys = [χy_fe([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    bs = [s.b([x, y, σ[i]], k_ws[i]) for i=1:nσ]
    ux = -differentiate(χys, z)
    uy = +differentiate(χxs, z)
    bz = differentiate(bs, z)

    fig, ax = plt.subplots(1, 3, figsize=(27*pc, 11.5*pc), sharey=true)

    axins1 = inset_locator.inset_axes(ax[1], width="50%", height="50%")
    ax[1].set_xlabel(L"Zonal flow $u^x$ $(\times 10^{-3})$")
    ax[2].set_xlabel(L"Meridional flow $u^y$")
    ax[3].set_xlabel(L"Stratification $\partial_z b$")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlim(-0.1, 1)
    ax[2].set_xlim(0, 0.2)
    ax[3].set_xlim(0, 1.3)
    ax[1].set_ylim(-H, 0)
    axins1.set_ylim(-H, -H + 0.05)
    for a ∈ ax
        a.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    end
    axins1.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)

    ax[1].plot(1e3*uy, z)
    axins1.plot(uy, z)
    ax[2].plot(-ux, z)
    ax[3].plot(bz, z)
    z = m2D.z[ix, :]
    ux = s2D.uξ[ix, :]
    uy = s2D.uη[ix, :]
    bz = 1/H*differentiate(s2D.b[ix, :], m2D.σ)
    ax[1].plot(1e3*ux, z, "k--", lw=0.5)
    axins1.plot(ux, z, "k--", lw=0.5)
    ax[2].plot(uy, z, "k--", lw=0.5, label="2D")
    ax[3].plot(bz, z, "k--", lw=0.5)
    ax[2].legend()
    savefig("$out_folder/profiles2Dvs3D.png")
    println("$out_folder/profiles2Dvs3D.png")
    plt.close()
end

compare_ux_uy_b(m, s, m2D, s2D, 0.0, 0.5)

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