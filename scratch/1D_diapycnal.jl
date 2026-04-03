using nuPGCM
using Printf
using PyPlot
using JLD2
using Roots 

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")

# params
α = 1/2^4     # aspect ratio
θ = atan(α) # slope angle
N² = 1/α    # background stratification
κ_B = 1e2   # κ bottom value
κ_I = 1     # κ interior value
h_κ = α/8   # κ decay scale

# output dir
dirname = joinpath(@__DIR__, "1d_model/diapycnal")
if !isdir(dirname)
    mkdir(dirname)
end

function diapycnal_transport_flat_iso()
    x = -exp.(range(log(4), -10, length=2^11))
    x = [x; 0]
    # x = range(-4, 0, length=2^12)
    @printf("        x_max = %.3e\n", maximum(x))
    @printf("        x_min = %.3e\n", minimum(x))
    @printf("    length(x) = %d\n", length(x))
    println("             ---")
    σϖ = @. -1/h_κ * (κ_B - κ_I) * exp(-(0 - α*x)/h_κ)
    T = -nuPGCM.trapz(σϖ, x)
    Ta = (κ_B - κ_I) / α
    @printf("            T = %.8e\n", T)
    @printf("           Ta = %.8e\n", Ta)
    @printf("     |T - Ta| = %.8e\n", abs(T - Ta))
    @printf("|T - Ta|/|Ta| = %.8e\n", abs(T - Ta)/abs(Ta))

    # α = 1/64
    # d = α/8
    # κ_B = 1e2
    # κ_I = 1
    # result: 6.33603267e+03
end

# diapycnal_transport_flat_iso()

function diapycnal_transport_b_exp(; h_b)
    b(x, z) = N²*z + h_b*N²*cos(θ)^2*exp(-(z - α*x)/h_b)
    zb(x) = find_zero(z -> b(x, z), 0)  # guess z = 0
    x_max = -h_b * cos(θ)^2 / α

    n = 2^12
    x = x_max .- exp.(range(10, -100, length=n))
    x = [x; x_max]
    @printf("        x_max = %.3e\n", maximum(x))
    @printf("        x_min = %.3e\n", minimum(x))
    @printf("    length(x) = %d\n", length(x))
    println("             ---")
    
    # filename = @sprintf("%s/integration_pts_a%d_hb%d.png", dirname, Int(1/α), Int(α/h_b))
    # fig, ax = plt.subplots(1)
    # xL = -8h_b/α
    # xp = range(xL, 0, length=2^8)
    # zp = range(xL*α, -xL*α, length=2^8)
    # ax.contour(xp, zp, [zp[i] > α*xp[j] ? b(xp[j], zp[i]) : NaN for i in eachindex(zp), j in eachindex(xp)],
    #     colors="k", alpha=1.0, linewidths=1.0, linestyles="-")
    # ax.spines["left"].set_visible(false)
    # ax.spines["bottom"].set_visible(false)
    # ax.plot(x, zb.(x), "C3.", ms=3)
    # ax.fill_between(xp, α*xp, xL*α*ones(2^8), facecolor="k", alpha=0.2)
    # ax.plot([x_max], α*[x_max], "C0o", ms=1)
    # ax.set_xlim(xL, 0)
    # ax.set_xlabel(L"x")
    # ax.set_ylabel(L"z")
    # savefig(filename)
    # @info "Saved '$filename'"
    # plt.close()

    x_flat = x .- x_max
    σϖ_flat = @. -α/h_κ * (κ_B - κ_I) * exp(-(0 - α*x_flat)/h_κ)

    b_z = @. N² - N²*cos(θ)^2*exp(-(zb(x) - α*x)/h_b)
    b_zz = @. 1/h_b * N²*cos(θ)^2 * exp(-(zb(x) - α*x)/h_b)

    κ = @. κ_I + (κ_B - κ_I) * exp(-(zb(x) - α*x)/h_κ)
    κ_z = @. -1/h_κ * (κ_B - κ_I) * exp(-(zb(x) - α*x)/h_κ)

    σϖ = @. α*(κ_z + κ*b_zz/b_z)

    T_flat = -nuPGCM.trapz(σϖ_flat, x_flat)
    T_exp = -nuPGCM.trapz(σϖ, x)
    @printf("T_flat = %.8e\n", T_flat)
    @printf("T_exp = %.8e\n", T_exp)

    # filename = @sprintf("%s/integrand_a%d_hb%d.png", dirname, Int(1/α), Int(α/h_b))
    filename = @sprintf("%s/integrand_a%d_hb%.1f.png", dirname, Int(1/α), h_b)
    fig, ax = plt.subplots(1)
    ax.fill_between(x_flat, -σϖ_flat, facecolor="gray", alpha=0.4, label=L"b' = 0")
    ax.fill_between(x,      -σϖ,      facecolor="C0",   alpha=0.7, label=L"b' \sim \exp")
    ax.plot(x, -α*κ_z,          "C3", lw=0.7, label=L"-\kappa_z")
    ax.plot(x, -α*κ.*b_zz./b_z, "C4", lw=0.7, label=L"-\kappa b_{zz} / b_z")
    # ax.plot(x, -κ,        lw=0.7, label=L"-\kappa")
    # ax.plot(x, -b_zz ./ (α*b_z),     lw=0.7, label=L"-b_{zz}/b_z")
    # ax.plot(x, -b_zz,     lw=0.7, label=L"-b_{zz}")
    # ax.plot(x, -1.0./b_z, lw=0.7, label=L"-1 / b_z")
    # ax.plot(x_max, κ_B,             "C0.", ms=1)
    # ax.plot(x_max, N²*cos(θ)^2/h_b, "C1.", ms=1)
    # ax.plot(x_max, 1/(N²*sin(θ)^2), "C2.", ms=1)
    # ax.plot(x_max, (κ_B-κ_I)/h_κ,   "C3.", ms=1)
    # ax.plot(x_max, -κ_B/h_b/α^2,    "C4.", ms=1)
    ax.set_xlim(-2, 0)
    # ax.set_xlim(x_max-2h_b, x_max)
    # ax.set_xlim(-1e-2, 0)
    ax.legend(loc=(1.05, 0.1))
    ax.set_yscale("symlog")
    # ymax = 10^round(log10(maximum(abs.(σϖ))), RoundUp)
    ymax = 10^7
    ax.set_ylim(-ymax, ymax)
    ax.set_yticks([-ymax, 0, ymax])
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"-\sigma\varpi")
    ax.text(0.05, 0.95, latexstring(@sprintf("\$T_{\\rm{flat}} = %s\$", nuPGCM.sci_notation(T_flat))), transform=ax.transAxes)
    ax.text(0.05, 0.85, latexstring(@sprintf("\$T_{\\rm{exp}} = %s\$",  nuPGCM.sci_notation(T_exp))),  transform=ax.transAxes)
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    return T_exp
end

T = diapycnal_transport_b_exp(; h_b=0.1)

# hs = α./2.0.^(0:1:10)
# Ts = [diapycnal_transport_b_exp(; h_b=h) for h in hs]