using nuPGCM
using nuPGCM.Numerics
using nuPGCM.OneDModel
using Printf
using PyPlot
using JLD2
using Roots
using Dierckx

pygui(false)
plt.style.use(joinpath(@__DIR__, "../plots.mplstyle"))
plt.close("all")


function load_parameters(; α)
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
    N² = 1/α
    θ = atan(α)
    f = 0.5
    Px = nothing
    U = 0
    Py = nothing
    V = 0
    H = α
    nz = 2^8
    # for eddy param:
    eddy_param = true
    νmin = κ₀/ν₀
    N²min = sqrt(1e-3)
    smoothing = 10

    z = H*OneDModel.chebyshev_nodes(nz)

    # κ needs to be in physical coordinates as it is in terms of HAB
    xz_phys = OneDModel.transform_to_physical.(0, z, θ)
    x_phys = first.(xz_phys)
    z_phys = last.(xz_phys)
    z_bot = z_phys[1] .+ α*(x_phys .- x_phys[1])
    h_κ = α/8
    κ_B = 1e2
    κ_I = 1
    κ = @. κ_I + (κ_B - κ_I)*exp(-(z_phys - z_bot)/h_κ)

    T = h_κ^2 / (κ_B*α^2*ε^2/μϱ)
    Δt = min(100*86400/t₀, T/100000)
    t_save = T/20
    @info "Time" T Δt t_save T÷Δt

    return (; μϱ, α, θ, ε, N², Δt, Px, U, Py, V, H, f, T, z, nz, κ, eddy_param, νmin, N²min, smoothing, t_save)
end

function setup_output(params)
    dirname = "1d_model/diapycnal_new_nu"
    label = @sprintf("_control_a%02d", Int(1/params.α))
    if params.eddy_param
        dirname *= "_eddy"
    end
    if !isdir(joinpath(@__DIR__, dirname))
        mkdir(joinpath(@__DIR__, dirname))
    end
    @info "Output directory: $(joinpath(@__DIR__, dirname))"
    @info "Label = '$label'"
    data_file = joinpath(@__DIR__, @sprintf("%s/sol_a%02d.jld2", dirname, Int(1/params.α))) 

    return dirname, label, data_file
end

function run_or_load_sim(params; data_file)
    if !isfile(data_file)
        # solve
        (; eddy_param, t_save) = params
        us, vs, Pxs, Pys, bs, ts = OneDModel.solve(params; eddy_param, t_save)
        @save data_file us vs Pxs Pys bs ts
        @info "Saved '$data_file'"
    else
        # load 
        @load data_file us vs Pxs Pys bs ts
        @info "Loaded '$data_file'"
    end
    return (; us, vs, Pxs, Pys, bs, ts)
end

function make_plots(params, sol; dirname, label)
    (; μϱ, α, θ, ε, N², Px, Py, H, f, z, nz, κ, eddy_param) = params
    (; us, vs, Pxs, Pys, bs, ts) = sol

    # BL thickness
    if eddy_param
        bz = differentiate(bs[:, end], z)
        ν = zeros(nz)
        OneDModel.update_ν!(ν, bs[:, end], params)
        ν_B = ν[1]
    else
        ν_B = 1
    end
    κ_B = κ[1]
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

"""
    calculate_diapycnal_transport()

Calculate T = ∫ σϖ dξ over an isopycnal.
"""
function calculate_diapycnal_transport(params, sol)
    (; α, ε, μϱ, θ, N², z, nz, κ) = params
    (; bs) = sol
    ź = z .+ α

    # select final timestep [note: this is b′(ź)]
    b′ = bs[:, end]  

    # choose isopycnal b₀ = N²z + b′
    b₀ = 0
    z₀ = b₀ / N² # for b′ = 0
    zL = (b₀ - b′[end])/N²
    zR = (b₀ - b′[1])/N²
    xL = zL/α - 1
    xR = zR/α

    # exponential grid confined near xR
    n = 2^10
    x = xR .- (xR - xL)*2.0.^(range(0, -52, length=n))
    x = [x; xR]

    # interpolate b′ so we can find roots
    b′_func = Spline1D(ź, b′)

    # function for z(x) along isopycnal
    ź_func(x, z) = OneDModel.transform_to_rotated(x, z, θ)[2]
    b(x, z) = N²*z + b′_func(ź_func(x, z))
    zb(x) = find_zero(z -> b(x, z), z₀)

    # arrays in physical coords
    nx = 10*nz
    x́2D = repeat(range(-1, 0, nx), 1, nz)
    ź2D = repeat(ź, 1, nx)'
    x2D = similar(x́2D)
    z2D = similar(ź2D)
    for i in 1:nx, j in 1:nz
        x2D[i, j], z2D[i, j] = OneDModel.transform_to_physical(x́2D[i, j], ź2D[i, j], θ)
    end
    b2D = N²*z2D + repeat(b′, 1, nx)'

    # plot isopycnals and b = b₀
    filename = joinpath(@__DIR__, @sprintf("%s/isopycnals_a%d_soln.png", dirname, Int(1/α)))
    fig, ax = subplots(1)
    levels = range(minimum(b2D), maximum(b2D), 20)
    ax.contour(x2D, z2D, b2D, levels=levels, linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
    ax.plot(x2D[:, 1], z2D[:, 1], "k-", lw=0.1)
    ax.contour(x2D, z2D, b2D, levels=[b₀], linestyles="-", colors="C0", linewidths=1)
    ax.plot(x, zb.(x), "C1-", lw=0.5)
    ax.set_xlabel(L"Horizontal coordinate $x$")
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.axis("equal")
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    # compute integrand as a function of ź
    b_z = N² .+ differentiate(b′, ź)*cos(θ)
    b_zz = differentiate(b_z, ź)*cos(θ)
    κ_z = differentiate(κ, ź)*cos(θ)
    σϖ = @. α^2*ε^2/μϱ * (κ_z + κ * b_zz / b_z)

    # interpolate, evaluate along isopycnal, and integrate
    σϖ_func = Spline1D(ź, σϖ)
    σϖ_iso = σϖ_func.(ź_func.(x, zb.(x)))
    T = nuPGCM.trapz(σϖ_iso, x)
    T_theory = α^2*ε^2/μϱ * κ[end]/α  # analytical solution
    @printf("T/α        = %.5e\n", T/α)
    @printf("T_theory/α = %.5e\n", T_theory/α)

    # plot integrand
    # x_flat = xL:0.01:xR_flat
    # κ_z_flat = Spline1D(ź, κ_z).(ź_func.(x_flat, z₀))
    κ_iso = Spline1D(ź, κ).(ź_func.(x, zb.(x)))
    κ_z_iso = Spline1D(ź, κ_z).(ź_func.(x, zb.(x)))
    b_z_iso = Spline1D(ź, b_z).(ź_func.(x, zb.(x)))
    b_zz_iso = Spline1D(ź, b_zz).(ź_func.(x, zb.(x)))
    filename = joinpath(@__DIR__, @sprintf("%s/integrand_a%d_soln.png", dirname, Int(1/α)))
    fig, ax = subplots(1)
    ax.fill_between(x, asinh.(σϖ_iso),   0,                                         label=L"\sigma\varpi")
    # ax.plot(x_flat,    asinh.(α^2*ε^2/μϱ * κ_z_flat),                 "C2", lw=0.7, label=L"\kappa_z(z = 0)")
    ax.plot(x,         asinh.(α^2*ε^2/μϱ * κ_z_iso),                  "C3", lw=0.7, label=L"\kappa_z")
    ax.plot(x,         asinh.(α^2*ε^2/μϱ * κ_iso.*b_zz_iso./b_z_iso), "C4", lw=0.7, label=L"\kappa b_{zz} / b_z")
    ax.set_xlabel(L"Horizontal coordinate $x$")
    ax.set_ylabel(L"$\sinh^{-1}$(integrand components)")
    ax.legend(loc="lower left")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-15, 15)
    # ax.set_ylim(-20, 20)
    ax.text(0.05, 0.95, latexstring(@sprintf("\$T/\\alpha = %s\$",                nuPGCM.sci_notation(T/α))), transform=ax.transAxes)
    ax.text(0.05, 0.85, latexstring(@sprintf("\$T_{\\rm{theory}}/\\alpha = %s\$", nuPGCM.sci_notation(T_theory/α))), transform=ax.transAxes)
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()

    return T
end

"""
    dVdt = calculate_volume_tendency(params, sol)

Calculate dV/dt ≈ ΔV/Δt where ΔV is the volume between the isopycnal b = b₀ at times t and t + Δt.
"""
function calculate_volume_tendency(params, sol)
    (; N², z, t_save, θ) = params
    (; bs) = sol
    ź = z
    Δt = t_save

    # select final two timesteps [note: this is b′(ź)]
    b′₁ = bs[:, end-1]
    b′₂ = bs[:, end]

    # choose isopycnal b₀ = N²x́*sin(θ) + N²*ź*cos(θ) + b′(t, ź)
    #   → x́ = (b₀ - b′(t, ź) - N²*ź*cos(θ)) / (N²*sin(θ))
    #   → Δx́ = (b′₁ - b′₂) / (N²*sin(θ))
    Δx́ = @. (b′₁ - b′₂) / (N²*sin(θ))

    # compute volume between isopycnals
    ΔV = nuPGCM.trapz(Δx́, ź)
    dVdt = ΔV / Δt
    @printf("dV/dt = %.3e\n", dVdt)

    return dVdt
end

function α_convergence()
    αs = 2.0.^-(2:8)
    Ts = zeros(length(αs))
    dVdts = zeros(length(αs))
    for (i, α) in enumerate(αs)
        params = load_parameters(; α)
        _, _, data_file = setup_output(params)
        sol = run_or_load_sim(params; data_file)
        Ts[i] = calculate_diapycnal_transport(params, sol)
        dVdts[i] = calculate_volume_tendency(params, sol)
    end

    # scale everything
    params = load_parameters(; α=αs[1])
    (; ε, μϱ) = params
    @. Ts *= μϱ / (αs * ε^2)
    @. dVdts *= μϱ / (αs * ε^2)

    filename = joinpath(@__DIR__, "$dirname/alpha_conv.png")
    fig, ax = plt.subplots(1)
    ax.axhline(-1, lw=0.5, ls=":", c="gray", label="Analytical")
    ax.axhline(0, lw=0.5, ls="-", c="k")
    ax.spines["bottom"].set_visible(false)
    ax.plot(log2.(αs), -Ts, "o", label=L"Diapycnal transport $-T$")
    ax.plot(log2.(αs), dVdts, "o", label=L"Volume tendency $\frac{\rm{d}V}{\rm{d}t}$")
    ax.legend()
    ax.set_xticks([-8, -6, -4, -2])
    ax.set_xticklabels([L"2^{-8}", L"2^{-6}", L"2^{-4}", L"2^{-2}"])
    ax.set_ylim(-2, 6)
    ax.set_xlabel(L"Aspect ratio $\alpha$")
    ax.set_ylabel(L"Value $\times \mu\varrho/\alpha\varepsilon^2$")
    savefig(filename) 
    @info "Saved '$filename'"
    plt.close()

    return αs, Ts, dVdts
end


# params = load_parameters(; α=2^-7)
# dirname, label, data_file = setup_output(params)
# sol = run_or_load_sim(params; data_file)
# make_plots(params, sol; dirname, label)
# # T = calculate_diapycnal_transport(params, sol)
# dVdt = calculate_volume_tendency(params, sol)

αs, Ts, dVdts = α_convergence()