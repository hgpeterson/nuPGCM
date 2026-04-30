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
ε = sqrt(ν₀/f₀/H₀^2)
μ = ν₀/κ₀
ϱ = (N₀*H₀/f₀/L)^2
t₀ = 1/f₀/ϱ  # s
μϱ = μ*ϱ
α = 1/2^5
N² = 1/α
θ = atan(α)
f = 0.5
Px = nothing
U = 0
Py = nothing
V = 0
H = α
nz = 2^8
eddy_param = true

z = H*OneDModel.chebyshev_nodes(nz)

# κ needs to be in physical coordinates as it is in terms of HAB
xz_phys = OneDModel.transform_to_physical.(0, z, θ)
x_phys = first.(xz_phys)
z_phys = last.(xz_phys)
z_bot = z_phys[1] .+ α*x_phys
h_κ = α/8
κ_B = 1e2
κ_I = 1
κ = @. κ_I + (κ_B - κ_I)*exp(-(z_phys - z_bot)/h_κ)

T = h_κ^2 / (κ_B*α^2*ε^2/μϱ)
# T = H^2 / (κ_B*α^2*ε^2/μϱ)
Δt = min(100*86400/t₀, T/100000)
t_save = T/20
@info "Time" T Δt t_save T÷Δt

params = (μϱ=μϱ, α=α, θ=θ, ε=ε, N²=N², Δt=Δt, Px=Px, U=U, Py=Py, V=V, H=H, f=f, T=T, z=z, nz=nz, κ=κ)

dirname = "1d_model/diapycnal"
label = @sprintf("_control_a%02d", Int(1/α))
if eddy_param
    dirname *= "_eddy"
end
if !isdir(joinpath(@__DIR__, dirname))
    mkdir(joinpath(@__DIR__, dirname))
end
@info "Saving in $(joinpath(@__DIR__, dirname))"
@info "Label = '$label'"

# solve
us, vs, Pxs, Pys, bs, ts = OneDModel.solve(params; eddy_param, t_save)
data_file = joinpath(@__DIR__, @sprintf("%s/sol_a%02d.jld2", dirname, Int(1/α))) 
@save data_file us vs Pxs Pys bs ts
@info "Saved '$data_file'"

# # load 
# @load data_file us vs Pxs Pys bs ts
# @info "Loaded '$data_file'"

function make_plots(; label="")
    z = params.z # ???

    # BL thickness
    if eddy_param
        bz = differentiate(bs[:, end], z)
        ν = zeros(nz)
        OneDModel.update_ν!(ν, bs[:, end], params)
        ν_B = ν[1]
    else
        ν_B = 1
    end
    κ_B = 1e2
    δ = α*ε*sqrt(2*ν_B/f)
    q = 1/δ * (1 + μϱ/α * ν_B/κ_B *  N²*tan(θ)^2 / f^2)^(1/4)
    @sprintf("BL scale q⁻¹ = %.3e", q^-1)

    # plot u, v, bz
    u = us[:, end]
    v = vs[:, end]
    Px = Pxs[end]
    Py = Pys[end]
    filename = joinpath(@__DIR__, "$dirname/profiles$label.png")
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
    ax[1].set_ylabel(latexstring(@sprintf("Vertical coordinate \$\\acute{z}/\\alpha\$ (\$\\alpha = 1/%d\$)", Int(1/α))))
    ax[1].set_xlabel("Flow")
    ax[2].set_xlabel(L"Stratification $\alpha (N^2 \cos \theta + \partial_{\acute z} b')$")
    for a ∈ ax
        a.set_ylim(-H, 0)
        a.spines["left"].set_visible(false)
        a.spines["top"].set_visible(true)
        a.axvline(0, color="k", lw=0.5)
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    end
    ax[2].set_yticks([])
    ax[1].plot(u,       z, "C0-", label=L"$\acute u$")
    ax[1].plot(v,       z, "C1-", label=L"$\acute v$")
    # ax[1].plot(+Px/f/cos(θ) .- b/α*sin(θ)/f/cos(θ), z, "C8--", label=L"$P_x/f' - \alpha^{-1} b \sin\theta / f'$")
    ax[1].axvline(-Py/f/cos(θ), c="C0", ls="--", lw=0.5, label=L"$-P_y/f'$")
    ax[1].axvline(+Px/f/cos(θ), c="C1", ls="--", lw=0.5, label=L"$P_x/f'$")
    # uvmax = maximum(abs.([u; v]))
    uvmax = 0.05
    ax[1].plot([-0.05*uvmax, 0.05*uvmax], [-H + q^-1, -H + q^-1], "C3-", lw=0.5)
    ax[1].set_xlim(-1.1*uvmax, 1.1*uvmax)
    ax[1].set_yticks([0, -H/2, -H])
    ax[1].set_yticklabels([L"0", L"-0.5", L"-1.0"])
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
    filename = joinpath(@__DIR__, "$dirname/nu$label.png")
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.set_ylabel(latexstring(@sprintf("Vertical coordinate \$\\acute{z}/\\alpha\$ (\$\\alpha = 1/%d\$)", Int(1/α))))
    ax.set_xlabel(L"Turbulent viscosity $\nu$")
    ax.set_xlim(0, 10)
    ax.set_ylim(-H, 0)
    ax.set_yticks([0, -H/2, -H])
    ax.set_yticklabels([L"0", L"-0.5", L"-1.0"])
    # ax.spines["left"].set_visible(false)
    # ax.axvline(0, color="k", lw=0.5)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    ν = ones(nz) # allocate
    for i in 2:size(bs, 2)
        alpha = 0.1 + 0.9*1.62^(i - size(bs, 2))
        OneDModel.update_ν!(ν, bs[:, i], params)
        ax.plot(ν, z, "k-", alpha=alpha)
    end
    ax.set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(ts[end]))))
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    # # plot bz (physical)
    # filename = joinpath(@__DIR__, "$dirname/bz$label.png")
    # fig, ax = plt.subplots(1, figsize=(2, 3.2))
    # ax.set_ylabel(latexstring(@sprintf("Vertical coordinate \$\\acute{z}/\\alpha\$ (\$\\alpha = 1/%d\$)", Int(1/α))))
    # ax.set_xlabel(L"Stratification $\alpha ( N^2 + \partial_{\acute z} b' \cos\theta)$")
    # # ax.set_xlim(0, 2)
    # ax.set_ylim(-H, 0)
    # ax.set_yticks([0, -H/2, -H])
    # ax.set_yticklabels([L"0", L"-0.5", L"-1.0"])
    # ax.spines["left"].set_visible(false)
    # ax.axvline(0, color="k", lw=0.5)
    # ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    # ν = ones(nz) # allocate
    # for i in 2:size(bs, 2)
    #     alpha = 0.1 + 0.9*1.62^(i - size(bs, 2))
    #     bz = α * ( N² .+ differentiate(bs[:, i], z)*cos(θ) )
    #     # bz = α^-1 * ( N² .+ differentiate(bs[:, i], z)*cos(θ) )
    #     ax.plot(bz, z, "k-", alpha=alpha)
    # end
    # ax.set_title(latexstring(@sprintf("\$t = %s\$", nuPGCM.sci_notation(ts[end]))))
    # savefig(filename)
    # @info "Saved '$filename'"
    # plt.close()

    # plot u, b over slope
    filename = joinpath(@__DIR__, "$dirname/slope$label.png")
    x́ = repeat(range(0, 1, nz), 1, nz)
    ź = repeat(z, 1, nz)'
    x, z = OneDModel.transform_to_physical(x́, ź, θ)
    b = bs[:, end]
    bb = N²*z + repeat(b, 1, nz)'
    uu = repeat(u, 1, nz)'*cos(θ)
    vmax = maximum(abs.(u))*cos(θ)
    fig, ax = subplots(1)
    img = ax.pcolormesh(x, z/α, uu, cmap="RdBu_r", rasterized=true, shading="auto", vmin=-vmax, vmax=vmax)
    cb = colorbar(img, ax=ax, label=L"Cross-slope flow $u$", shrink=0.5)
    # cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    levels = range(minimum(bb), maximum(bb), 20)
    ax.contour(x, z/α, bb, levels=levels, linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
    ax.set_xlabel(L"Horizontal coordinate $x$")
    ax.set_ylabel(latexstring(@sprintf("Vertical coordinate \$z/\\alpha\$\n(\$\\alpha = 1/%d\$)", Int(1/α))))
    # ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 0])
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()
end

# make_plots(; label)

"""
    calculate_diapycnal_transport()

Calculate -∫ σϖ dξ over an isopycnal.
"""
function calculate_diapycnal_transport()
    # physical coords
    z = params.z # ???
    nx = 10*nz
    x́ = repeat(range(0, 1, nx), 1, nz)
    ź = repeat(z, 1, nx)'
    x = similar(x́)
    z = similar(ź)
    for i in 1:nx, j in 1:nz
        x[i, j], z[i, j] = OneDModel.transform_to_physical(x́[i, j], ź[i, j], θ)
    end

    # mixing array
    κ = repeat(params.κ, 1, nx)'

    # isopycnals from solution
    b = N²*z + repeat(bs[:, end], 1, nx)'
    b₀ = 0
    j_iso = [argmin(abs.(b[i, :] .- b₀)) for i=1:nx]
    i_mask = findall(i -> j_iso[i] > 1, 1:nx)
    x_iso = [x[i, j_iso[i]] for i in i_mask]
    z_iso = [z[i, j_iso[i]] for i in i_mask]

    # plot isopycnals and b = b₀
    filename = joinpath(@__DIR__, @sprintf("%s/isopycnals_a%d_soln.png", dirname, Int(1/α)))
    fig, ax = subplots(1)
    levels = range(minimum(b), maximum(b), 20)
    ax.contour(x, z, b, levels=levels, linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
    ax.plot(x[:, 1], z[:, 1], "k-", lw=0.1)
    ax.contour(x, z, b, levels=[b₀], linestyles="-", colors="C0", linewidths=1)
    ax.plot(x_iso, z_iso, "C1-", lw=0.5)
    ax.set_xlabel(L"Horizontal coordinate $x$")
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.axis("equal")
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    # compute integrand (and components)
    κ_z = zeros(nx, nz)
    b_z = zeros(nx, nz)
    b_zz = zeros(nx, nz)
    for i in 1:nx
        b_z[i, :] = differentiate(b[i, :], z[i, :])
        b_zz[i, :] = differentiate(b_z[i, :], z[i, :])
        κ_z[i, :] = differentiate(κ[i, :], z[i, :])
    end
    # σϖ = α * ∂z(κ ∂z(b)) / ∂z(b) = α * ( ∂z(κ) + κ ∂zz(b) / ∂z(b) )
    σϖ = @. α * ( κ_z + κ * b_zz / b_z )

    # plot 2D integrand
    filename = joinpath(@__DIR__, @sprintf("%s/integrand_2D_a%d_soln.png", dirname, Int(1/α)))
    fig, ax = subplots(1)
    vmax = 1e4
    img = ax.pcolormesh(x, z/α, σϖ, cmap="RdBu_r", rasterized=true, shading="auto", vmin=-vmax, vmax=vmax)
    colorbar(img, ax=ax, label=L"Thickness $\times$ diapycnal flow $\sigma\varpi$", extend="both", shrink=0.5)
    levels = range(minimum(b), maximum(b), 20)
    ax.contour(x, z/α, b, levels=levels, linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
    ax.plot(x[:, 1], z[:, 1]/α, "k-")
    ax.contour(x, z/α, b, levels=[b₀], linestyles="-", colors="C3", linewidths=1.0, alpha=0.5)
    ax.set_xlabel(L"Horizontal coordinate $x$")
    ax.set_ylabel(latexstring(@sprintf("Vertical coordinate \$z/\\alpha\$\n(\$\\alpha = 1/%d\$)", Int(1/α))))
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 0])
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    # compute integral over isopycnal
    σϖ_iso = [σϖ[i, j_iso[i]] for i in i_mask]
    T = -nuPGCM.trapz(σϖ_iso, x_iso)
    T_flat = κ_B - κ_I  # analytical solution to -∫ κ_z dx for x ∈ (-∞, 0] at z = 0
    @printf("T      = %.3e\n", T)
    @printf("T_flat = %.3e\n", T_flat)

    # plot integrand
    κ_iso = [κ[i, j_iso[i]] for i in i_mask]
    κ_z_iso = [κ_z[i, j_iso[i]] for i in i_mask]
    b_z_iso = [b_z[i, j_iso[i]] for i in i_mask]
    b_zz_iso = [b_zz[i, j_iso[i]] for i in i_mask]
    j_0 = [argmin(abs.(z[i, :])) for i in 1:nx]  # z = 0
    κ_z_0 = [κ_z[i, j_0[i]] for i in 1:nx]
    x_0 = [x[i, j_0[i]] for i in 1:nx]
    filename = joinpath(@__DIR__, @sprintf("%s/integrand_a%d_soln.png", dirname, Int(1/α)))
    fig, ax = subplots(1)
    ax.fill_between(x_iso, -asinh.(σϖ_iso), 0, label=L"\sigma\varpi")
    ax.plot(x_0,   -asinh.(α*κ_z_0),                    "C2", lw=0.7, label=L"\alpha\kappa_z(z = 0)")
    ax.plot(x_iso, -asinh.(α*κ_z_iso),                  "C3", lw=0.7, label=L"\alpha\kappa_z")
    ax.plot(x_iso, -asinh.(α*κ_iso.*b_zz_iso./b_z_iso), "C4", lw=0.7, label=L"\alpha\kappa b_{zz} / b_z")
    ax.set_xlabel(L"Horizontal coordinate $x$")
    ax.set_ylabel(L"$-\sinh^{-1}$(integrand components)")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(-15, 15)
    ax.text(0.05, 0.95, latexstring(@sprintf("\$T = %s\$",              nuPGCM.sci_notation(T))), transform=ax.transAxes)
    ax.text(0.05, 0.85, latexstring(@sprintf("\$T_{\\rm{flat}} = %s\$", nuPGCM.sci_notation(T_flat))), transform=ax.transAxes)
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()
end

calculate_diapycnal_transport()