using nuPGCM
using nuPGCM.Numerics
using nuPGCM.OneDModel
using Printf
using PyPlot
using JLD2

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

# parameters

# μϱ = 1
# α = 1/4
# θ = atan(2α)
# ε = 0.1
# N² = 1/α
# no_Px = false
# no_Py = false
# H = α
# f = 1
# nz = 2^8
# T = μϱ/ε^2/1e2
# Δt = 1e-4*T
# eddy_param = false

Ω = 2π/86400  # s⁻¹
a = 6.371e6  # m
β = 2Ω/a  # m⁻¹ s⁻¹
L = 2π*a*60/360  # m
f₀ = β*L  # s⁻¹
H₀ = 4e3  # m
κ₀ = 1e-5  # m² s⁻¹
Kₑ = 1000  # m² s⁻¹
N₀ = 1e-3  # s⁻¹
ν₀ = Kₑ*f₀^2/N₀^2  # m² s⁻¹
ν₀ /= 16
# κ₀ /= 32
ε = sqrt(ν₀/f₀/H₀^2)
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2
t₀ = 1/f₀/ϱ  # s
μϱ = μ*ϱ
α = 1/4
N² = 1/α
θ = atan(2α)
f = 0.5
Px = nothing
U = 0
Py = nothing
V = 0
H = α*cos(θ)
nz = 2^8
eddy_param = false

z = H*OneDModel.chebyshev_nodes(nz)
d = H/8
κ_B = 1e2
κ_I = 1
κ = @. κ_I + (κ_B - κ_I)exp(-(z + H)/d)

# T = d^2 / (κ_B*α^2*ε^2/μϱ)
T = H^2 / (κ_B*α^2*ε^2/μϱ)
Δt = min(100*86400/t₀, T/100000)
t_save = T/20
@info "Time" T Δt t_save T÷Δt

params = (μϱ=μϱ, α=α, θ=θ, ε=ε, N²=N², Δt=Δt, Px=Px, U=U, Py=Py, V=V, H=H, f=f, T=T, z=z, nz=nz, κ=κ)

dirname = "1d_model/Py_test"
if eddy_param
    dirname *= "_eddy"
end
if !isdir(joinpath(@__DIR__, dirname))
    mkdir(joinpath(@__DIR__, dirname))
end
@info "Saving in $(joinpath(@__DIR__, dirname))"

# solve
us, vs, Pxs, Pys, bs, ts = OneDModel.solve(params; eddy_param, t_save)

function make_plots()
    z = params.z # ???

    # BL thickness
    if eddy_param
        bz = differentiate(bs[:, end], z)
        ν = abs.(f^2 * cos(θ)^2 ./ ( α * (N² .+ cos(θ)*bz) ))
        ν[ν .> 1e2] .= 1e2
        ν_B = ν[1]
    else
        ν_B = 1
    end
    κ_B = 1e2
    δ = α*ε*sqrt(2*ν_B/f)
    q = 1/δ * (1 + 1/α * ν_B/κ_B * μ * N²*tan(θ) / f^2 * ϱ)^(1/4)
    @sprintf("BL scale q⁻¹ = %.3e", q^-1)

    # plot u, v, bz
    u = us[:, end]
    v = vs[:, end]
    Px = Pxs[end]
    Py = Pys[end]
    filename = joinpath(@__DIR__, "$dirname/profiles.png")
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel("Flow")
    ax[2].set_xlabel(L"Stratification $\alpha (N^2 \cos \theta + \partial_z b)$")
    for a ∈ ax
        a.set_ylim(-H, 0)
        a.spines["left"].set_visible(false)
        a.spines["top"].set_visible(true)
        a.axvline(0, color="k", lw=0.5)
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    end
    ax[2].set_yticks([])
    ax[1].plot(u,       z, "C0-", label=L"$u$")
    ax[1].plot(v,       z, "C1-", label=L"$v$")
    # ax[1].plot(+Px/f/cos(θ) .- b/α*sin(θ)/f/cos(θ), z, "C8--", label=L"$P_x/f' - \alpha^{-1} b \sin\theta / f'$")
    ax[1].axvline(-Py/f/cos(θ), c="C0", ls="--", lw=0.5, label=L"$-P_y/f'$")
    ax[1].axvline(+Px/f/cos(θ), c="C1", ls="--", lw=0.5, label=L"$P_x/f'$")
    uvmax = maximum(abs.([u; v]))
    # ax[1].plot([-0.05*uvmax, 0.05*uvmax], [-H + q^-1, -H + q^-1], "C3-", lw=0.5)
    # ax[1].set_xlim(-1.1*uvmax, 1.1*uvmax)
    ax[1].plot([-0.05*0.1, 0.05*0.1], [-H + q^-1, -H + q^-1], "C3-", lw=0.5)
    ax[1].set_xlim(-0.1, 0.1)
    ax[1].set_yticks([0, -H/2, -H])
    ax[1].legend(loc="upper left")
    for i in 2:size(bs, 2)
        alpha = 0.1 + 0.9*1.62^(i - size(bs, 2))
        bz = differentiate(bs[:, i], z)
        ax[2].plot(α*(N²*cos(θ) .+ bz), z, "k-", alpha=alpha)
    end
    ax[2].set_xlim(-0.2, 1.3)
    ax[1].set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(ts[end]))))
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    # plot ν
    filename = joinpath(@__DIR__, "$dirname/nu.png")
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.set_xlabel(L"Turbulent viscosity $\nu$")
    ax.set_ylim(-H, 0)
    ax.set_yticks([0, -H/2, -H])
    ax.spines["left"].set_visible(false)
    ax.axvline(0, color="k", lw=0.5)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    for i in 2:size(bs, 2)
        alpha = 0.1 + 0.9*1.62^(i - size(bs, 2))
        bz = differentiate(bs[:, i], z)
        ν = abs.(f^2 * cos(θ)^2 ./ ( α * (N² .+ cos(θ)*bz) ))
        ν[ν .> 1e2] .= 1e2
        ax.plot(ν, z, "k-", alpha=alpha)
    end
    ax.set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(ts[end]))))
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    # plot u, b over slope
    filename = joinpath(@__DIR__, "$dirname/slope.png")
    x′ = repeat(range(0, 0.5, nz), 1, nz)
    z′ = repeat(z, 1, nz)'
    x = x′*cos(θ) - z′*sin(θ)
    z = x′*sin(θ) + z′*cos(θ)
    b = bs[:, end]
    bb = N²*z + repeat(b, 1, nz)'
    uu = repeat(u, 1, nz)'*cos(θ)
    vmax = maximum(abs.(u))*cos(θ)
    fig, ax = subplots(1)
    img = ax.pcolormesh(x, z, uu, cmap="RdBu_r", rasterized=true, shading="auto", vmin=-vmax, vmax=vmax)
    cb = colorbar(img, ax=ax, label=L"Cross-slope flow $u$", shrink=0.5)
    # cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    levels = range(minimum(bb), maximum(bb), 20)
    ax.contour(x, z, bb, levels=levels, linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks([0, 0.5])
    ax.set_yticks([-α, 0])
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()
end

make_plots()