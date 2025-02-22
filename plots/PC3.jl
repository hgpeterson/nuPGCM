using nuPGCM
using PyPlot
using PyCall
using JLD2
using Printf

# full width (39 picas) for three or more panels across
# just under full width (33 picas, which is 85% of full width) for two panels across
# two-thirds page width (27 picas) for single panel figures that have detail or text that needs to be larger than single column width
# 19 picas (single column width). 

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
    cb = fig.colorbar(img, ax=ax[4], label=L"Barotropic streamfunction $\Psi$"*"\n"*L"(\times 10^{-2})", fraction=1.0)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)

    # save
    savefig("psi.png")
    @info "Saved 'psi.png'"
    savefig("psi.pdf")
    @info "Saved 'psi.pdf'"
    plt.close()
end

function slices(field)
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
        j = argmin(abs.(y)) # index where y = 0
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
        if field == "u" 
            vmax = 1.8
        elseif field == "v"
            vmax = 7.2
        elseif field == "w"
            vmax = 2.8
        end
        img = ax[i].pcolormesh(xx, z, 1e2*f, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
        ax[i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        if i == 1
            ax[i].plot([0.5, 0.5], [-0.75, 0.0], "r-", alpha=0.7)
        end
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

function profiles()
    width = 33pc
    fig, ax = plt.subplots(1, 4, figsize=(width, width/4*1.62))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[4].annotate("(d)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"Zonal flow $u$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Meridional flow $v$"*"\n"*L"($\times 10^{-2}$)")
    ax[3].set_xlabel(L"Vertical flow $w$"*"\n"*L"($\times 10^{-2}$)")
    ax[4].set_xlabel(L"Stratification $\partial_z b$")
    ax[1].set_xlim(-1.5, 1.5)
    ax[2].set_xlim(-4.5, 4.5)
    ax[3].set_xlim(-1.5, 1.5)
    ax[4].set_xlim(0, 1.1)
    ax[1].spines["left"].set_visible(false)
    ax[2].spines["left"].set_visible(false)
    ax[3].spines["left"].set_visible(false)
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[4].set_yticks([])
    ax[1].axvline(0, color="k", lw=0.5)
    ax[2].axvline(0, color="k", lw=0.5)
    ax[3].axvline(0, color="k", lw=0.5)
    for a ∈ ax 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
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
        b = d["b"]
        close(d)
        j = argmin(abs.(x .- 0.5)) # index where x = 0.5
        k = argmin(abs.(y)) # index where y = 0
        u = u[j, k, :]; u[1] = 0
        v = v[j, k, :]; v[1] = 0
        w = w[j, k, :]; w[1] = 0; w[end] = 0
        b = b[j, k, :]; b[end] = 0
        z = H[j, k]*σ
        Bz = 1 .+ differentiate(b, z)
        Bz[1] = 0
        umask = isnan.(u) .== 0
        vmask = isnan.(v) .== 0
        wmask = isnan.(w) .== 0
        Bzmask = isnan.(Bz) .== 0
        ax[1].plot(1e2*u[umask],  z[umask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[2].plot(1e2*v[vmask],  z[vmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[3].plot(1e2*w[wmask],  z[wmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
        ax[4].plot(Bz[Bzmask],    z[Bzmask], label=latexstring(@sprintf("\$\\beta = %0.1f\$", βs[i])))
    end
    # 1D U = 0
    file = jldopen("../sims/sim048/data/1D_beta0.0.jld2")
    u = file["u"]
    v = file["v"]
    w = file["w"]
    b = file["b"]
    z = file["z"]
    close(file)
    Bz = 1 .+ differentiate(b, z)
    ax[1].plot(1e2*u,  z, "k--", lw=0.5, label=L"1D $U=0$")
    ax[2].plot(1e2*v,  z, "k--", lw=0.5, label=L"1D $U=0$")
    ax[3].plot(1e2*w,  z, "k--", lw=0.5, label=L"1D $U=0$")
    ax[4].plot(Bz,     z, "k--", lw=0.5, label=L"1D $U=0$")

    # 1D U = V = 0
    file = jldopen("../sims/sim048/data/1D_beta1.0.jld2")
    u = file["u"]
    v = file["v"]
    w = file["w"]
    b = file["b"]
    z = file["z"]
    close(file)
    Bz = 1 .+ differentiate(b, z)
    ax[1].plot(1e2*u,  z, "k-.", lw=0.5, label=L"1D $U=V=0$")
    ax[2].plot(1e2*v,  z, "k-.", lw=0.5, label=L"1D $U=V=0$")
    ax[3].plot(1e2*w,  z, "k-.", lw=0.5, label=L"1D $U=V=0$")
    ax[4].plot(Bz,     z, "k-.", lw=0.5, label=L"1D $U=V=0$")

    ax[2].legend(loc=(-0.6, 0.5))
    savefig("profiles.png")
    @info "Saved 'profiles.png'"
    savefig("profiles.pdf")
    @info "Saved 'profiles.pdf'"
    plt.close()
end

function alpha()
    width = 27pc
    fig, ax = plt.subplots(1, 3, figsize=(width, width/3*1.62), sharey=true)
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"Cross-slope flow $u$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Along-slope flow $v$"*"\n"*L"($\times 10^{-2}$)")
    ax[3].set_xlabel(L"Stratification $\partial_z b$")
    ax[1].spines["left"].set_visible(false)
    ax[2].spines["left"].set_visible(false)
    ax[1].axvline(0, color="k", lw=0.5)
    ax[2].axvline(0, color="k", lw=0.5)
    αs = 0:0.1:1
    colors = pl.cm.viridis(range(0, 1, length=length(αs)))
    for i ∈ eachindex(αs)
        file = jldopen(@sprintf("../scratch/data/1D_%0.2f.jld2", αs[i]))
        u = file["u"]
        v = file["v"]
        b = file["b"]
        z = file["z"]
        bz = differentiate(b, z)
        ax[1].plot(1e2*u,   z, c=colors[i, :], label=latexstring(@sprintf("\$\\alpha = %0.2f\$", αs[i])))
        ax[2].plot(1e2*v,   z, c=colors[i, :], label=latexstring(@sprintf("\$\\alpha = %0.2f\$", αs[i])))
        ax[3].plot(1 .+ bz, z, c=colors[i, :], label=latexstring(@sprintf("\$\\alpha = %0.2f\$", αs[i])))
        close(file)
    end
    ax[1].legend(loc=(0.3, 0.12))
    savefig("alpha.png")
    println("alpha.png")
    savefig("alpha.pdf")
    println("alpha.pdf")
    plt.close()
end


# f_over_H()
psi()
# slices("u")
# slices("v")
# slices("w")
# zonal_sections()
# profiles()
# alpha()

println("Done.")