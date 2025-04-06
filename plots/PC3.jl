using nuPGCM
using PyPlot
using PyCall
using JLD2
using Printf

include("baroclinic.jl")

# full width (39 picas)
# just under full width (33 picas)
# two-thirds page width (27 picas)
# single column width (19 picas)

pl = pyimport("matplotlib.pylab")
cm = pyimport("matplotlib.cm")
colors = pyimport("matplotlib.colors")

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

# pica
pc = 1/6

### helper functions 

include("derivatives.jl")

function fill_nan(f, i, j)
    sum = 0
    count = 0
    for m ∈ i-1:i+1, n ∈ j-1:j+1
        if m ≥ 1 && n ≥ 1 && m ≤ size(f, 1) && n ≤ size(f, 2)
            if !isnan(f[m, n])
                sum += f[m, n]
                count += 1
            end
        end
    end
    return sum / count
end

function fill_nans!(f)
    ff = copy(f)
    for i ∈ axes(f, 1), j ∈ axes(f, 2)
        if isnan(f[i, j])
            ff[i, j] = fill_nan(f, i, j)
        end
    end
    f .= ff
    return f
end

function compute_bbot(b, z)
    if !isnan(b[1])
        return b[1]
    elseif z[2] == z[1] == 0 # H = 0
        return 0
    else
        # have to extrapolate if NaN
        i1 = 2
        while isnan(b[i1])
            i1 += 1
        end
        i2 = i1 + 1
        while isnan(b[i2])
            i2 += 1
        end
        return b[i1] + (b[i2] - b[i1])/(z[i2] - z[i1])*(z[1] - z[i1])
    end
end

function compute_bx(b, x, σ, H)
    Hx = differentiate(H, x)

    # bx = bξ - σ Hx/H bσ
    bx = zeros(size(b))
    for j in axes(b, 2)
        bx[:, j] = differentiate(b[:, j], x)
    end
    for i in axes(b, 1)
        if H[i] != 0
            bx[i, :] -= Hx[i]/H[i]*σ.*differentiate(b[i, :], σ)
        end
    end
    return bx
end

### figure creation functions

function f_over_H()
    # params/funcs
    f₀ = 1
    H(x, y) = 1 - x^2 - y^2
    f_over_H(x, y; β = 0) = (f₀ + β*y) / (H(x, y) + eps())
    vmax = 6

    # circular grid
    p, t = get_p_t("../meshes/circle.msh")
    x = p[:, 1]
    y = p[:, 2]
    t = t .- 1

    # setup
    fig, ax = plt.subplots(1, 4, figsize=(33pc, 11pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1].set_title(L"\beta = 0")
    ax[2].set_title(L"\beta = 0.5")
    ax[3].set_title(L"\beta = 1")
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    for a ∈ ax
        a.set_xlabel(L"Zonal coordinate $x$")
        a.axis("equal")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:1)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 1.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    ax[1].set_ylabel(L"Meridional coordinate $y$")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_visible(false)

    # plot
    levels = (1:4)/4 * vmax
    cmap = "GnBu"
    z = f_over_H.(x, y, β=0.0)
    ax[1].tripcolor(x, y, t, z, cmap=cmap, shading="gouraud", rasterized=true, vmin=0, vmax=vmax)
    ax[1].tricontour(x, y, t, z, levels=levels, colors="k", linewidths=0.5)
    ax[1].plot([-1.0, 1.0], [0.0, 0.0], "r-", alpha=0.7)
    ax[1].plot([0.5], [0.0], "ro", alpha=0.7, ms=2)
    z = f_over_H.(x, y, β=0.5)
    ax[2].tripcolor(x, y, t, z, cmap=cmap, shading="gouraud", rasterized=true, vmin=0, vmax=vmax)
    ax[2].tricontour(x, y, t, z, levels=levels, colors="k", linewidths=0.5)
    z = f_over_H.(x, y, β=1.0)
    img = ax[3].tripcolor(x, y, t, z, cmap=cmap, shading="gouraud", rasterized=true, vmin=0, vmax=vmax)
    ax[3].tricontour(x, y, t, z, levels=levels, colors="k", linewidths=0.5)
    cb = fig.colorbar(img, ax=ax[4], label=L"Planetary vorticity $f/H$", extend="max", fraction=1.0)

    # save
    savefig("f_over_H.png")
    println("f_over_H.png") 
    savefig("f_over_H.pdf")
    println("f_over_H.pdf") 
    plt.close()
end

function buoyancy()
    fig, ax = plt.subplots(1, 2, figsize=(19pc, 8pc), gridspec_kw=Dict("width_ratios"=>[1, 2]))

    # (a) buoyancy profiles
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_xlabel(L"Stratification $\partial_z b$")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlim(0, 1.3)
    ax[1].set_yticks(-0.75:0.25:0)
    ts = 5e-3:5e-3:5e-2
    colors = pl.cm.BuPu_r(range(0, 0.8, length=length(ts)))
    for i in eachindex(ts)
        file = jldopen(@sprintf("../sims/sim048/data/1D_b_%1.1e.jld2", ts[i]), "r")
        b = file["b"]
        z = file["z"]
        close(file)
        bz = differentiate(b, z)
        ax[1].plot(1 .+ bz, z, c=colors[i, :])
    end
    ax[1].annotate("", xy=(0.6, -0.57), xytext=(0.42, -0.42), arrowprops=Dict("color"=>"k", "arrowstyle"=>"-|>"), fontsize=6)
    ax[1].annotate(L"$t = 5 \times 10^{-2}$", xy=(0.1, -0.4), fontsize=6)

    # (b) buoyancy gradient
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].set_xlabel(L"Zonal coordinate $x$")
    ax[2].set_ylabel(L"Vertical coordinate $z$")
    ax[2].axis("equal")
    ax[2].set_xticks([0, 1])
    ax[2].set_yticks([-1, 0])
    ax[2].spines["left"].set_visible(false)
    ax[2].spines["bottom"].set_visible(false)
    d = jldopen("buoyancy.jld2")
    x = d["x"]
    z = d["z"]
    bx = d["bx"]
    close(d)
    xx = repeat(x, 1, size(z, 2))
    vmax = maximum(abs.(bx))
    img = ax[2].pcolormesh(xx, z, bx, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
    ax[2].contour(xx, z, z .+ b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
    ax[2].plot([0.5, 0.5], [-0.75, 0.0], "r-", alpha=0.7)
    cb = fig.colorbar(img, ax=ax[2], label=L"Buoyancy gradient $\partial_x b$")
    # cb.set_ticks([-vmax, 0, vmax])

    subplots_adjust(wspace=0.5)

    savefig("buoyancy.png")
    @info "Saved 'buoyancy.png'"
    savefig("buoyancy.pdf")
    @info "Saved 'buoyancy.pdf'"
    plt.close()
end

function psi()
    # params/funcs
    f₀ = 1
    H(x, y) = 1 - x^2 - y^2
    f_over_H(x, y; β = 0) = (f₀ + β*y) / (H(x, y) + eps())
    f_over_H_levels = (1:4)/4 * 6
    get_levels(vmax) = [-vmax, -3vmax/4, -vmax/2, -vmax/4, vmax/4, vmax/2, 3vmax/4, vmax]

    # setup
    fig, ax = plt.subplots(1, 4, figsize=(33pc, 11pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1].set_title(L"\beta = 0")
    ax[2].set_title(L"\beta = 0.5")
    ax[3].set_title(L"\beta = 1")
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    for a ∈ ax
        a.set_xlabel(L"Zonal coordinate $x$")
        a.axis("equal")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:1)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 1.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    ax[1].set_ylabel(L"Meridional coordinate $y$")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_visible(false)

    # plot
    βs = [0.0, 0.5, 1.0]
    img = nothing
    vmax = 0.
    for i ∈ eachindex(βs)
        d = jldopen(@sprintf("../sims/sim048/data/psi_beta%1.1f_i003.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        psi = d["psi"]
        close(d)
        vmax = nuPGCM.nan_max(abs.(psi))
        @info "vmax = $vmax"
        vmax = 1.5
        img = ax[i].pcolormesh(x, y, psi', shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
        ax[i].contour(x, y, 1e2*psi', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(vmax))
        foH = [f_over_H(x[j], y[k], β=βs[i]) for j ∈ eachindex(x), k ∈ eachindex(y)]
        ax[i].contour(x, y, foH', colors=(0.2, 0.5, 0.2), linewidths=0.5, alpha=0.5, linestyles="-", levels=f_over_H_levels)
    end
    fig.colorbar(img, ax=ax[4], label=L"Barotropic streamfunction $\Psi$"*"\n"*L"(\times 10^{-2})", fraction=1.0)

    # save
    savefig("psi.png")
    @info "Saved 'psi.png'"
    savefig("psi.pdf")
    @info "Saved 'psi.pdf'"
    plt.close()
end

function zonal_sections_single_field(field)
    # setup
    fig, ax = plt.subplots(1, 4, figsize=(39pc, 8pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1].set_title(L"\beta = 0")
    ax[2].set_title(L"\beta = 0.5")
    ax[3].set_title(L"\beta = 1")
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    for a ∈ ax
        a.set_xlabel(L"Zonal coordinate $x$")
        a.axis("equal")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:0)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 0.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[2].set_yticklabels([])
    ax[3].set_yticklabels([])
    ax[4].set_visible(false)

    # plot
    βs = [0.0, 0.5, 1.0]
    img = nothing
    vmax = 0.
    for i ∈ eachindex(βs)
        # load gridded sigma data
        d = jldopen(@sprintf("../sims/sim048/data/gridded_sigma_beta%1.1f_n0257_i003.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        σ = d["σ"]
        H = d["H"]
        f = d[field]
        b = d["b"]
        close(d)
        xx = repeat(x, 1, length(σ))
        # j = argmin(abs.(y)) # index where y = 0
        y0 = √3 - 2
        j = argmin(abs.(y .- y0))
        z = H[:, j]*σ'
        f = f[:, j, :]
        fill_nans!(f)
        f[:, 1] .= 0
        if "field" == "w"
            f[:, end] .= 0
        end
        b = z .+ b[:, j, :]
        fill_nans!(b)
        b[:, end] .= 0
        # if field == "u" 
        #     vmax = 1.8
        # elseif field == "v"
        #     vmax = 7.2
        # elseif field == "w"
        #     vmax = 2.8
        # end
        vmax = 1e2*nan_max(abs.(f))
        println(vmax)
        img = ax[i].pcolormesh(xx, z, 1e2*f, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
        ax[i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        # if i == 1
        #     ax[i].plot([0.5, 0.5], [-0.75, 0.0], "r-", alpha=0.7)
        # end
    end
    if field == "u"
        label = L"Zonal flow $u$"*"\n"*L"($\times 10^{-2}$)"
    elseif field == "v"
        label = L"Meridional flow $v$"*"\n"*L"($\times 10^{-2}$)"
    elseif field == "w"
        label = L"Vertical flow $w$"*"\n"*L"($\times 10^{-2}$)"
    end
    fig.colorbar(img, ax=ax[4], label=label, fraction=1.0)

    # save
    savefig("$field.png")
    @info "Saved '$field.png'"
    savefig("$field.pdf")
    @info "Saved '$field.pdf'"
    plt.close()
end

function zonal_sections_single_sim(β)
    # Load data
    d = jldopen(@sprintf("../sims/sim048/data/gridded_sigma_beta%1.1f_n0257_i003.jld2", β))
    x = d["x"]
    y = d["y"]
    σ = d["σ"]
    H = d["H"]
    u = d["u"]
    v = d["v"]
    w = d["w"]
    b = d["b"]
    close(d)

    # Prepare data
    xx = repeat(x, 1, length(σ))
    # j = argmin(abs.(y)) # index where y = 0
    y0 = √3 - 2
    j = argmin(abs.(y .- y0))
    z = H[:, j] * σ'
    u = u[:, j, :]
    v = v[:, j, :]
    w = w[:, j, :]
    fill_nans!(u)
    fill_nans!(v)
    fill_nans!(w)
    u[:, 1] .= 0
    v[:, 1] .= 0
    w[:, 1] .= 0
    w[:, end] .= 0
    b = z .+ b[:, j, :]
    fill_nans!(b)
    b[:, end] .= 0

    # Plot setup
    fig, ax = plt.subplots(1, 3, figsize=(39pc, 12pc), gridspec_kw=Dict("width_ratios" => [1, 1, 1], "wspace" => 0.3))
    components = [(u, L"Zonal flow $u$"), (v, L"Meridional flow $v$"), (w, L"Vertical flow $w$")]
    vmax_values = [1.8, 7.2, 2.8]  # Adjust these values as needed for colorbar limits

    for i ∈ 1:3
        f, label = components[i]
        vmax = vmax_values[i]
        img = ax[i].pcolormesh(xx, z, 1e2 * f, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
        ax[i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        ax[i].set_xlabel(L"Zonal coordinate $x$")
        # ax[i].set_title(label)
        ax[i].axis("equal")
        ax[i].set_xticks(-1:1:1)
        ax[i].set_yticks(-1:1:0)
        ax[i].set_xlim(-1.05, 1.05)
        ax[i].set_ylim(-1.05, 0.05)
        ax[i].spines["left"].set_visible(false)
        ax[i].spines["bottom"].set_visible(false)
        if i > 1
            ax[i].set_yticklabels([])
        else
            ax[i].set_ylabel(L"Vertical coordinate $z$")
        end
        cb = fig.colorbar(img, ax=ax[i], orientation="horizontal", pad=0.3, fraction=0.05)
        cb.set_label(label * L" ($\times 10^{-2}$)")
    end

    # Save the figure
    ofile = @sprintf("zonal_sections_beta%1.1f", β)
    savefig(ofile * ".png")
    @info "Saved '$ofile.png'"
    savefig(ofile * ".pdf")
    @info "Saved '$ofile.pdf'"
    plt.close()
end

function zonal_sections()
    # setup
    fig, ax = plt.subplots(3, 4, figsize=(39pc, 24pc), gridspec_kw=Dict("width_ratios"=>[1, 1, 1, 0.05]))
    ax[1, 1].set_title(L"\beta = 0")
    ax[1, 2].set_title(L"\beta = 0.5")
    ax[1, 3].set_title(L"\beta = 1")
    ax[1, 1].annotate("(a)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[1, 3].annotate("(c)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[2, 1].annotate("(d)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[2, 2].annotate("(e)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[2, 3].annotate("(f)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[3, 1].annotate("(g)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[3, 2].annotate("(h)", xy=(-0.00, 0.95), xycoords="axes fraction")
    ax[3, 3].annotate("(i)", xy=(-0.00, 0.95), xycoords="axes fraction")
    for a ∈ ax
        a.axis("equal")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:0)
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 0.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
    end
    ax[3, 1].set_xlabel(L"Zonal coordinate $x$")
    ax[3, 2].set_xlabel(L"Zonal coordinate $x$")
    ax[3, 3].set_xlabel(L"Zonal coordinate $x$")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[3, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 1].set_xticklabels([])
    ax[1, 2].set_xticklabels([])
    ax[1, 3].set_xticklabels([])
    ax[2, 1].set_xticklabels([])
    ax[2, 2].set_xticklabels([])
    ax[2, 3].set_xticklabels([])
    ax[1, 2].set_yticklabels([])
    ax[2, 2].set_yticklabels([])
    ax[3, 2].set_yticklabels([])
    ax[1, 3].set_yticklabels([])
    ax[2, 3].set_yticklabels([])
    ax[3, 3].set_yticklabels([])
    ax[1, 4].set_visible(false)
    ax[2, 4].set_visible(false)
    ax[3, 4].set_visible(false)

    # plot
    βs = [0.0, 0.5, 1.0]
    umax = 1.8
    vmax = 7.2
    wmax = 2.8
    for i ∈ eachindex(βs)
        # load gridded sigma data
        d = jldopen(@sprintf("../sims/sim048/data/gridded_sigma_beta%1.1f_n0257_i003.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        σ = d["σ"]
        H = d["H"]
        u = d["u"]
        v = d["v"]
        w = d["w"]
        b = d["b"]
        close(d)
        xx = repeat(x, 1, length(σ))
        j = argmin(abs.(y)) # index where y = 0
        z = H[:, j]*σ'
        u = u[:, j, :]
        v = v[:, j, :]
        w = w[:, j, :]
        fill_nans!(u)
        fill_nans!(v)
        fill_nans!(w)
        u[:, 1] .= 0
        v[:, 1] .= 0
        w[:, 1] .= 0
        w[:, end] .= 0
        b = z .+ b[:, j, :]
        fill_nans!(b)
        b[:, end] .= 0
        @info "vmax values" maximum(abs.(u)) maximum(abs.(v)) maximum(abs.(w))
        ax[1, i].pcolormesh(xx, z, 1e2*u, shading="gouraud", cmap="RdBu_r", vmin=-umax, vmax=umax, rasterized=true)
        ax[1, i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        ax[2, i].pcolormesh(xx, z, 1e2*v, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
        ax[2, i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        ax[3, i].pcolormesh(xx, z, 1e2*w, shading="gouraud", cmap="RdBu_r", vmin=-wmax, vmax=wmax, rasterized=true)
        ax[3, i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
    end
    ax[1, 1].plot([0.5, 0.5], [-0.75, 0.0], "r-", alpha=0.7)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-umax, vmax=umax), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax[1, 4], label=L"Zonal flow $u$"*"\n"*L"($\times 10^{-2}$)", fraction=1.0)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-vmax, vmax=vmax), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax[2, 4], label=L"Meridional flow $v$"*"\n"*L"($\times 10^{-2}$)", fraction=1.0)
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=-wmax, vmax=wmax), cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax[3, 4], label=L"Vertical flow $w$"*"\n"*L"($\times 10^{-2}$)", fraction=1.0)

    # save
    savefig("zonal_sections.png")
    @info "Saved 'zonal_sections.png'"
    savefig("zonal_sections.pdf")
    @info "Saved 'zonal_sections.pdf'"
    plt.close()
end

function flow_profiles()
    width = 27pc
    fig, ax = plt.subplots(1, 3, figsize=(width, width/3*1.62))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"Zonal flow $u$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Meridional flow $v$"*"\n"*L"($\times 10^{-2}$)")
    ax[3].set_xlabel(L"Vertical flow $w$"*"\n"*L"($\times 10^{-2}$)")
    ax[1].set_xlim(-1.5, 1.5)
    ax[2].set_xlim(-4.5, 4.5)
    ax[3].set_xlim(-1.5, 1.5)
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    for a ∈ ax 
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", lw=0.5)
    end
    βs = [0.0, 0.5, 1.0]
    for i ∈ eachindex(βs)
        # load gridded sigma data
        d = jldopen(@sprintf("../sims/sim048/data/gridded_sigma_beta%1.1f_n0257_i003.jld2", βs[i]))
        x = d["x"]
        y = d["y"]
        σ = d["σ"]
        H = d["H"]
        u = d["u"]
        v = d["v"]
        w = d["w"]
        close(d)
        j = argmin(abs.(x .- 0.5)) # index where x = 0.5
        k = argmin(abs.(y)) # index where y = 0
        u = u[j, k, :]; u[1] = 0
        v = v[j, k, :]; v[1] = 0
        w = w[j, k, :]; w[1] = 0; w[end] = 0
        z = H[j, k]*σ
        umask = isnan.(u) .== 0
        vmask = isnan.(v) .== 0
        wmask = isnan.(w) .== 0
        ax[1].plot(1e2*u[umask], z[umask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[2].plot(1e2*v[vmask], z[vmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[3].plot(1e2*w[wmask], z[wmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
    end

    # # 1D U = 0
    # file = jldopen("../sims/sim048/data/1D_beta0.0.jld2")
    # u = file["u"]
    # v = file["v"]
    # w = file["w"]
    # z = file["z"]
    # close(file)
    # ax[1].plot(1e2*u, z, "k--", lw=0.5, label=L"1D $U=0$")
    # ax[2].plot(1e2*v, z, "k--", lw=0.5, label=L"1D $U=0$")
    # ax[3].plot(1e2*w, z, "k--", lw=0.5, label=L"1D $U=0$")

    # # 1D U = V = 0
    # file = jldopen("../sims/sim048/data/1D_beta1.0.jld2")
    # u = file["u"]
    # v = file["v"]
    # w = file["w"]
    # z = file["z"]
    # close(file)
    # ax[1].plot(1e2*u, z, "k-.", lw=0.5, label=L"1D $U=V=0$")
    # ax[2].plot(1e2*v, z, "k-.", lw=0.5, label=L"1D $U=V=0$")
    # ax[3].plot(1e2*w, z, "k-.", lw=0.5, label=L"1D $U=V=0$")

    # # get bx from 3D model
    # d = jldopen("../sims/sim048/data/gridded_sigma_beta0.0_n0257_i003.jld2")
    # x = d["x"]
    # y = d["y"]
    # σ = d["σ"]
    # H = d["H"]
    # b = d["b"]
    # close(d)
    # j0 = length(y) ÷ 2 + 1 # index where y = 0
    # H = H[:, j0]
    # b = b[:, j0, :]
    # zz = H*σ'
    # b[:, end] .= 0 # b = 0 at z = 0
    # b[end, :] .= 0 # b = 0 where H = 0
    # b[:, 1] .= [compute_bbot(b[i, :], zz[i, :]) for i in eachindex(x)] # fill nans at z = -H
    # fill_nans!(b) # everywhere else
    # bx = compute_bx(b, x, σ, H)
    # i0 = argmin(abs.(x .- 0.5)) # index where x = 0.5
    # bx = bx[i0, :]
    # z = zz[i0, :]
    # nz = length(z)

    # get bx from 1D model
    d = jldopen(@sprintf("../sims/sim048/data/1D_b_%1.1e.jld2", 3e-3), "r")
    b = d["b"]
    z = d["z"]
    close(d)
    nz = length(z)
    bx = -differentiate(b, z)

    # get BL solutions
    u, v, w = solve_baroclinic_problem_BL_U0(ε=1e-2, z=z, ν=ones(nz), f=1, bx=bx, Hx=-1)
    ax[1].plot(1e2*u, z, "k--", lw=0.5, label=L"$U = 0$ theory")
    ax[2].plot(1e2*v, z, "k--", lw=0.5, label=L"$U = 0$ theory")
    ax[3].plot(1e2*w, z, "k--", lw=0.5, label=L"$U = 0$ theory")
    u, v, w = solve_baroclinic_problem_BL(ε=1e-2, z=z, ν=ones(nz), f=1, β=1, bx=bx, by=zeros(nz), U=0, V=0, τx=0, τy=0, Hx=-1, Hy=0)
    # u, v, w = solve_baroclinic_problem_BL(ε=1e-2, z=z, ν=ones(nz), f=1, β=1, bx=bx, by=zeros(nz), U=-4.1e-4, V=1.5e-3, τx=0, τy=0, Hx=-1, Hy=0)
    ax[1].plot(1e2*u, z, "k-.", lw=0.5, label=L"$U = V = 0$ theory")
    ax[2].plot(1e2*v, z, "k-.", lw=0.5, label=L"$U = V = 0$ theory")
    ax[3].plot(1e2*w, z, "k-.", lw=0.5, label=L"$U = V = 0$ theory")

    ax[2].legend(loc=(-0.6, 0.5))
    savefig("flow_profiles.png")
    @info "Saved 'flow_profiles.png'"
    savefig("flow_profiles.pdf")
    @info "Saved 'flow_profiles.pdf'"
    plt.close()

    # fig, ax = plt.subplots(1, figsize=(2, 3.2))
    # ax.set_xlabel(L"\partial_x b")
    # ax.set_ylabel(L"Vertical coordinate $z$")
    # ax.plot(bx3D, z3D)
    # ax.plot(bx1D, z1D)
    # ax.spines["left"].set_visible(false)
    # ax.axvline(0, color="k", lw=0.5)
    # ax.set_yticks([-0.75, -0.5, -0.25, 0])
    # savefig("bx.png")
    # @info "Saved 'bx.png'"
    # plt.close()
end

function psi_bl()
    # # load b from 3D model
    # d = jldopen("../sims/sim048/data/gridded_sigma_beta0.0_n0257_i003.jld2", "r")
    # b = d["b"]
    # x = d["x"]
    # y = d["y"]
    # σ = d["σ"]
    # H = d["H"]
    # close(d)

    # load Ψ from 3D model
    d = jldopen("../sims/sim048/data/psi_beta0.0_n0257_003.jld2", "r")
    x3D = d["x"]
    Ψ3D = d["Ψ"]
    close(d)

    # slice at y = 0 from x = 0 to 1
    i0 = size(Ψ3D, 1)÷2 + 1
    j0 = size(Ψ3D, 2)÷2 + 1
    x3D = x3D[i0:end]
    Ψ3D = Ψ3D[i0:end, j0]
    # b = b[i0:end, j0, :]
    # Ψ = Ψ[i0:end, j0]
    # H = H[i0:end, j0]
    # z = H*σ'
    # # b[:, 1] .= [compute_bbot(b[i, :], H[i]*σ) for i in eachindex(x)]
    # b[:, end] .= 0 # b = 0 at z = 0
    # b[end, :] .= 0 # b = 0 where H = 0
    # b[:, 1] .= [compute_bbot(b[i, :], z[i, :]) for i in eachindex(x)] # fill nans at z = -H
    # fill_nans!(b) # everywhere else
    # bx = compute_bx(b, x, σ, H)
    d = jldopen("buoyancy.jld2", "r")
    x = d["x"]
    z = d["z"]
    bx = d["bx"]
    close(d)

    # compute Ψ from BL theory
    ε = 1e-2
    α = 1/2
    V_BL_1 = compute_V_BL(bx, x, z, ε, α; order=1)
    Ψ_BL_1 = cumtrapz(V_BL_1, x) .- trapz(V_BL_1, x)
    V_BL_2 = compute_V_BL(bx, x, z, ε, α; order=2)
    Ψ_BL_2 = cumtrapz(V_BL_2, x) .- trapz(V_BL_2, x)

    # plot
    fig, ax = plt.subplots(1)
    ax.set_xlim(0, 1)
    ax.set_xticks(0:0.5:1)
    ax.set_ylim(-6, 0)
    ax.set_yticks(-6:2:0)
    ax.spines["bottom"].set_position("zero")
    ax.xaxis.set_label_coords(0.5, 1.25)
    ax.tick_params(axis="x", top=true, labeltop=true, bottom=false, labelbottom=false)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.plot(x3D, 1e2*Ψ3D,    "C0",          label="3D model")
    ax.plot(x,   1e2*Ψ_BL_1, "k-",  lw=0.5, label=L"BL theory to $O(1)$")
    ax.plot(x,   1e2*Ψ_BL_2, "k--", lw=0.5, label=L"BL theory to $O(\varepsilon)$")
    ax.legend()
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Barotropic streamfunction $\Psi$ ($\times 10^{-2}$)")
    savefig("psi_bl.png")
    @info "Saved 'psi_bl.png'"
    savefig("psi_bl.pdf")
    @info "Saved 'psi_bl.pdf'"
    plt.close()
end
function compute_V_BL(bx, x, z, ε, α; order)
    if !(order in [1, 2, 3])
        throw(ArgumentError("Invalid `order`: $order; must be 1, 2, or 3."))
    end
    
    # parameters
    # q = 1/√2
    H = -z[:, 1]
    Hx = differentiate(H, x)
    Γ = @. 1 + α^2*Hx^2
    q = @. Γ^(-3/4)/√2

    # order 1
    V = [-trapz(bx[i, :].*z[i, :], z[i, :]) for i in axes(bx, 1)]
    order -= 1

    if order != 0
        # order 2
        V -= @. ε*H/q*bx[:, 1]
        order -= 1
    end

    if order != 0
        # order 3
        bxz_bot = zeros(size(bx, 1))
        for i in 1:size(bx, 1)-1
            bxz_bot[i] = differentiate_pointwise(bx[i, 1:3], z[i, 1:3], z[i, 1], 1) 
        end
        V -= @. ε^2*(H*bxz_bot - bx[:, 1])/(2q^2)
        order -= 1
    end

    # if all went well, order should be 0
    @assert order == 0

    return V
end

function alpha()
    width = 19pc
    fig, ax = plt.subplots(1, 2, figsize=(width, width/2*1.62))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"Cross-slope flow $u$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Along-slope flow $v$"*"\n"*L"($\times 10^{-2}$)")
    ax[1].spines["left"].set_visible(false)
    ax[2].spines["left"].set_visible(false)
    ax[1].axvline(0, color="k", lw=0.5)
    ax[2].axvline(0, color="k", lw=0.5)
    ax[1].set_yticks(-0.75:0.25:0)
    ax[2].set_yticks([])
    ax[1].set_xlim(-0.2, 0.45)
    ax[2].set_xlim(-1, 4.5)
    # α = 0
    file = jldopen("../scratch/data/1D_0.00.jld2")
    u = file["u"]
    v = file["v"]
    z = file["z"]
    ax[1].plot(1e2*u, z, label=L"\alpha = 0")
    ax[2].plot(1e2*v, z, label=L"\alpha = 0")
    close(file)
    # α = 1/2
    file = jldopen("../scratch/data/1D_0.50.jld2")
    u = file["u"]
    v = file["v"]
    z = file["z"]
    ax[1].plot(1e2*u, z, label=L"\alpha = 1/2")
    ax[2].plot(1e2*v, z, label=L"\alpha = 1/2")
    close(file)
    ax[1].legend(loc=(0.45, 0.55))
    savefig("alpha.png")
    println("alpha.png")
    savefig("alpha.pdf")
    println("alpha.pdf")
    plt.close()
end


# f_over_H()
# buoyancy()
# psi()
# zonal_sections_single_field("u")
# zonal_sections_single_field("v")
# zonal_sections_single_field("w")
# zonal_sections_single_sim(0.0)
# zonal_sections_single_sim(0.5)
# zonal_sections_single_sim(1.0)
# zonal_sections()
# flow_profiles()
psi_bl()
# alpha()

println("Done.")