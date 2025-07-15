using nuPGCM
using JLD2
using Printf
using PyPlot
using PyCall

# include("../meshes/mesh_bowl2D.jl")

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

pc = 1/6 
mpl = pyimport("matplotlib")
pl = pyimport("matplotlib.pylab")
Line2D = pyimport("matplotlib.lines").Line2D
Poly3DCollection = pyimport("mpl_toolkits.mplot3d.art3d").Poly3DCollection

set_out_dir!(".")

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

function mesh()
    α = 1/2
    h = α/10
    if !isfile(@sprintf("../meshes/bowl2D_%e_%e.msh", h, α))
        generate_bowl_mesh_2D(h, α)
        mv(@sprintf("bowl2D_%e_%e.msh", h, α), @sprintf("../meshes/bowl2D_%e_%e.msh", h, α))
    end
    p, t = get_p_t(@sprintf("../meshes/bowl2D_%e_%e.msh", h, α))

    fig, ax = plt.subplots(1)
    ax.tripcolor(p[:, 1], p[:, 3], t .- 1, 0*t[:, 1], cmap="Greys", edgecolors="k", lw=0.2, rasterized=true)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-α, 0])
    ax.set_yticklabels([L"-\alpha", L"0"])
    savefig(@sprintf("%s/images/mesh.svg", out_dir))
    println(@sprintf("%s/images/mesh.svg", out_dir))
    plt.close()
end

function convergence()
    d = jldopen("../scratch/data/errors2D.jld2", "r")
    Es_2D = d["δu_H1"] .+ d["δp_L2"]
    E∞s_2D = d["δu_L∞"]
    dims_2D = d["dims"]
    εs_2D = d["εs"]
    αs_2D = d["αs"]
    hs_2D = d["hs"]
    close(d)

    d = jldopen("../scratch/data/errors3D_1e-6.jld2", "r")
    Es_3D = d["δu_H1"] .+ d["δp_L2"]
    E∞s_3D = d["δu_L∞"]
    dims_3D = d["dims"]
    εs_3D = d["εs"]
    αs_3D = d["αs"]
    hs_3D = d["hs"]
    close(d)

    d = jldopen("../scratch/data/errors3D_1e-7.jld2", "r")
    Es_3D_e7 = d["δu_H1"] .+ d["δp_L2"]
    E∞s_3D_e7 = d["δu_L∞"]
    close(d)

    d = jldopen("../scratch/data/errors3D_1e-8.jld2", "r")
    Es_3D_e8 = d["δu_H1"] .+ d["δp_L2"]
    E∞s_3D_e8 = d["δu_L∞"]
    close(d)

    colors = Dict(1/2=>"C0", 1/4=>"C1") # colors for α
    markers = Dict(1e-1=>"o", 1e-2=>"^") # markers for ε

    fig, ax = plt.subplots(2, 2, figsize=(33pc, 33pc))

    ax[1, 1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1, 2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2, 1].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2, 2].annotate("(d)", xy=(-0.04, 1.05), xycoords="axes fraction")

    ax[1, 1].set_title("2D")
    ax[1, 2].set_title("3D")

    for a in ax
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel(L"Resolution $h$")
    end

    ax[1, 1].set_ylabel(L"Energy norm error $||\mathbf{u}_h||_{H^1} + ||p_h - p||_{L^2}$")
    ax[1, 2].set_ylabel(L"Energy norm error $||\mathbf{u}_h||_{H^1} + ||p_h - p||_{L^2}$")
    ax[2, 1].set_ylabel(L"Max norm error $||\mathbf{u}_h||_{L^\infty}$")
    ax[2, 2].set_ylabel(L"Max norm error $||\mathbf{u}_h||_{L^\infty}$")

    ax[1, 1].set_xlim(1e-3, 4e-2)
    ax[2, 1].set_xlim(1e-3, 4e-2)
    ax[1, 2].set_xlim(5e-3, 5e-2)
    ax[2, 2].set_xlim(5e-3, 5e-2)
    ax[1, 2].set_xticklabels([L"10^{-2}"], minor="off")
    ax[2, 2].set_xticklabels([L"10^{-2}"], minor="off")

    ax[1, 1].set_ylim(1e-7, 1e1)
    ax[1, 2].set_ylim(1e-3, 1e2)
    ax[2, 1].set_ylim(1e-8, 1e0)
    ax[2, 2].set_ylim(1e-5, 1e0)

    # (a)
    for i in eachindex(dims_2D), j in eachindex(εs_2D), k in eachindex(αs_2D)
        if αs_2D[k] == 1 || εs_2D[j] == 1e0
            continue
        end
        ax[1, 1].plot(hs_2D, Es_2D[i, j, k, :], "-", c=colors[αs_2D[k]], marker=markers[εs_2D[j]])
    end
    h1, h2 = 2e-3, 2e-2
    ax[1, 1].plot([h1, h2], 5e-7/h1^2*[h1^2, h2^2], "k-")
    ax[1, 1].text(x=h2/3, y=1e-7/h1^2*(h2/3)^2, s=L"$h^2$")

    # (b)
    for i in eachindex(dims_3D), j in eachindex(εs_3D), k in eachindex(αs_3D)
        if εs_3D[j] == 1e0
            continue
        elseif εs_3D[j] == 1e-1 && αs_3D[k] == 1/2
            err = Es_3D_e7[1, 1, 1, :]
            err[1] = Es_3D_e8[1, 1, 1, 1]
            ax[1, 2].plot(hs_3D, err, "-", c=colors[αs_3D[k]], marker=markers[εs_3D[j]])
            continue
        end
        ax[1, 2].plot(hs_3D, Es_3D[i, j, k, :], "-", c=colors[αs_3D[k]], marker=markers[εs_3D[j]])
    end
    h1, h2 = 1e-2, 2e-2
    ax[1, 2].plot([h1, h2], 2e-3/h1^2*[h1^2, h2^2], "k-")
    ax[1, 2].text(x=1.4e-2, y=1e-3/h1^2*(1.4e-2)^2, s=L"$h^2$")

    # (c)
    for i in eachindex(dims_2D), j in eachindex(εs_2D), k in eachindex(αs_2D)
        if αs_2D[k] == 1 || εs_2D[j] == 1e0
            continue
        end
        ax[2, 1].plot(hs_2D, E∞s_2D[i, j, k, :], "-", c=colors[αs_2D[k]], marker=markers[εs_2D[j]])
    end
    h1, h2 = 2e-3, 2e-2
    ax[2, 1].plot([h1, h2], 2e-8/h1^3*[h1^3, h2^3], "k-")
    ax[2, 1].text(x=h2/3, y=5e-9/h1^3*(h2/3)^3, s=L"$h^3$")

    # (d)
    for i in eachindex(dims_3D), j in eachindex(εs_3D), k in eachindex(αs_3D)
        if εs_3D[j] == 1e0
            continue
        elseif εs_3D[j] == 1e-1 && αs_3D[k] == 1/2
            err = E∞s_3D_e7[1, 1, 1, :]
            err[1] = E∞s_3D_e8[1, 1, 1, 1]
            ax[2, 2].plot(hs_3D, err, "-", c=colors[αs_3D[k]], marker=markers[εs_3D[j]])
            continue
        end
        ax[2, 2].plot(hs_3D, E∞s_3D[i, j, k, :], "-", c=colors[αs_3D[k]], marker=markers[εs_3D[j]])
    end
    h1, h2 = 1e-2, 2e-2
    ax[2, 2].plot([h1, h2], 2e-5/h1^3*[h1^3, h2^3], "k-")
    ax[2, 2].text(x=1.4e-2, y=8e-6/h1^3*(1.4e-2)^3, s=L"$h^3$")

    custom_handles = [Line2D([0], [0], color="k",  marker=markers[1e-2], linestyle=""),
                      Line2D([0], [0], color="k",  marker=markers[1e-1], linestyle=""),
                      Line2D([0], [0], color=colors[1/4], linestyle="-"),
                      Line2D([0], [0], color=colors[1/2], linestyle="-")]
    custom_labels = [L"$\varepsilon = 10^{-2}$", L"$\varepsilon = 10^{-1}$",
                     L"$\alpha = 1/4$", L"$\alpha = 1/2$",] 
    ax[1, 2].legend(custom_handles, custom_labels, loc=(0.75, 0.5))

    subplots_adjust(wspace=0.4, hspace=0.4)

    savefig(@sprintf("%s/images/convergence.png", out_dir))
    println(@sprintf("%s/images/convergence.png", out_dir))
    savefig(@sprintf("%s/images/convergence.pdf", out_dir))
    println(@sprintf("%s/images/convergence.pdf", out_dir))
    plt.close()
end

function zonal_sections()
    # parameters
    α = 1/2

    # load data
    d = jldopen(@sprintf("../sims/sim050/data/gridded_n0257_t%016d.jld2", 250))
    # d = jldopen(@sprintf("../sims/sim051/data/gridded_n0257_t%016d.jld2", 250))
    x = d["x"]
    y = d["y"]
    σ = d["σ"]
    H = d["H"]
    u = d["u"]
    v = d["v"]
    w = d["w"]
    b = d["b"]
    close(d)

    # prepare data
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
    b = z/α .+ b[:, j, :]
    fill_nans!(b)
    b[:, end] .= 0

    fig, ax = plt.subplots(1, 3, figsize=(33pc, 8pc))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    components = [(u, L"Zonal flow $u_1$"), (v, L"Meridional flow $u_2$"), (w, L"Vertical flow $u_3$")]
    vmax_values = [0.42, 2.7, 0.27] # f-plane
    # vmax_values = [0.57, 2.3, 0.33] # β-plane
    # println(1e2*[maximum(abs.(u)), maximum(abs.(v)), maximum(abs.(w))])
    for i ∈ 1:3
        f, label = components[i]
        vmax = vmax_values[i]
        img = ax[i].pcolormesh(xx, z, 1e2 * f, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
        ax[i].contour(xx, z, b, colors="k", linewidths=0.5, alpha=0.3, linestyles="-", levels=-0.9:0.1:-0.1)
        ax[i].set_xlabel(L"Zonal coordinate $x_1$")
        ax[i].set_xticks(-1:1:1)
        ax[i].set_xlim(-1.05, 1.05)
        ax[i].set_ylim(-1.05, 0.05)
        ax[i].axis("equal")
        ax[i].spines["left"].set_visible(false)
        ax[i].spines["bottom"].set_visible(false)
        if i > 1
            ax[i].set_yticklabels([])
            ax[i].set_yticks([])
        else
            ax[i].set_ylabel(L"Vertical coordinate $x_3$")
            ax[i].set_yticks([-0.5, 0])
            ax[i].plot([0.5, 0.5], [-0.75*α, 0], "r-", alpha=0.7)
        end
        cb = fig.colorbar(img, ax=ax[i], orientation="horizontal", pad=0.4, fraction=0.05)
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_label(label * L" ($\times 10^{-2}$)")
    end
    savefig("images/zonal_sections_beta0.png")
    println("images/zonal_sections_beta0.png")
    savefig("images/zonal_sections_beta0.pdf")
    println("images/zonal_sections_beta0.pdf")
    plt.close()
end

function vertical_sections()
    # aspect ratio
    α = 1/2

    # load data
    d = jldopen(@sprintf("../sims/sim051/data/gridded_n0257_t%016d.jld2", 250))
    x = d["x"]
    y = d["y"]
    σ = d["σ"]
    H = d["H"]
    u = d["u"]
    v = d["v"]
    w = d["w"]
    b = d["b"]
    close(d)

    # plot setup
    fig, ax = plt.subplots(1, 2, figsize=(27pc, 27pc/1.62/2))
    for a in ax
        a.axis("equal")
        a.set_xlim(-1.05, 1.05)
        a.set_ylim(-1.05, 1.05)
        a.spines["left"].set_visible(false)
        a.spines["bottom"].set_visible(false)
        a.set_xlabel(L"Zonal coordinate $x_1$")
        a.set_ylabel(L"Meridional coordinate $x_2$")
        a.set_xticks(-1:1:1)
        a.set_yticks(-1:1:1)
    end
    ax[2].set_ylabel("")
    ax[2].set_yticklabels([])
    ax[2].set_yticks([])
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")

    # surface velocity
    u_sfc = u[:, :, end]
    v_sfc = v[:, :, end]
    speed = @. sqrt(u_sfc^2 + v_sfc^2)
    cb_max = 2.0
    arrow_length = 0.1 # inches
    scale = cb_max/1e2/arrow_length
    scale_units = "inches"
    img = ax[1].pcolormesh(x, y, 1e2*speed', shading="nearest", cmap="viridis", vmin=0, vmax=cb_max, rasterized=true) # need to use nearest for NaNs
    cb = plt.colorbar(img, ax=ax[1], label=L"Surface speed $\sqrt{u_1^2 + u_2^2}$ ($\times 10^{-2}$)", extend="max")
    n = length(x)
    slice = 1:2^3:n
    ax[1].quiver(x[slice], y[slice], u_sfc[slice, slice]', v_sfc[slice, slice]', color="w", pivot="mid", scale=scale, scale_units=scale_units)
    ax[1].plot(cos.(0:0.01:2π), sin.(0:0.01:2π), "k-", lw=1)

    # buoyancy at z = -α/2
    H[H .< 0] .= NaN
    z = [H[i, j]*σ[k] for i in 1:length(x), j in 1:length(y), k in eachindex(σ)]
    ks = [argmin(abs.(z[i, j, :] .+ α/2)) for i in 1:length(x), j in 1:length(y)]
    b = [-0.5 + b[i, j, ks[i, j]] for i in 1:length(x), j in 1:length(y)]
    b[end-10:end, :] .= NaN
    fill_nans!(b)
    vmin = -0.51
    vmax = -0.47
    img = ax[2].pcolormesh(x, y, b', shading="nearest", cmap="plasma", vmin=vmin, vmax=vmax, rasterized=true)
    cb = fig.colorbar(img, ax=ax[2], label=L"Buoyancy $b$ at $x_3/\alpha = -1/2$")
    cb.set_ticks([vmin, (vmin + vmax)/2, vmax])
    ax[2].plot(cos.(0:0.01:2π), sin.(0:0.01:2π), "k-", lw=0.5, alpha=0.5)

    subplots_adjust(wspace=0.4)

    savefig("images/vertical_sections.png")
    println("images/vertical_sections.png")
    savefig("images/vertical_sections.pdf")
    println("images/vertical_sections.pdf")
    plt.close()
end

function aspect_ratios()
    t = 25 # time

    # load data
    fname = @sprintf("../sims/sim049/alpha2/data/profile_%016d.jld2", t*10)
    profile2 = jldopen(fname)["profile2"]
    @info "Loaded '$fname'."

    fname = @sprintf("../sims/sim049/alpha4/data/profile_%016d.jld2", t*100)
    profile4 = jldopen(fname)["profile4"]
    @info "Loaded '$fname'."

    fname = @sprintf("../sims/sim049/alpha8/data/profile_%016d.jld2", t*200)
    profile8 = jldopen(fname)["profile8"]
    @info "Loaded '$fname'."

    fname = @sprintf("../sims/sim050/data/profile_%016d.jld2", t*10)
    profile2_3D = jldopen(fname)["profile2_3D"]
    @info "Loaded '$fname'."

    # collect
    profiles2D = [profile2, profile4, profile8]
    profiles3D = [profile2_3D]
    αs = [1/2, 1/4, 1/8, 1/2]

    # plot
    fig, ax = plt.subplots(1, 4, figsize=(33pc, 33pc/4*1.62))
    ax[1].annotate("(a)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[2].annotate("(b)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[3].annotate("(c)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[4].annotate("(d)", xy=(-0.04, 1.05), xycoords="axes fraction")
    ax[1].set_ylabel(L"Vertical coordinate $x_3 / \alpha$")
    ax[1].set_xlabel(L"Zonal flow $u_1$"*"\n"*L"($\times 10^{-2}$)")
    ax[2].set_xlabel(L"Meridional flow $u_2$"*"\n"*L"($\times 10^{-2}$)")
    ax[3].set_xlabel(L"Vertical flow $u_3 / \alpha$"*"\n"*L"($\times 10^{-2}$)")
    ax[4].set_xlabel(L"Stratification $\alpha \partial_{x_3} b$")
    ax[1].set_xlim(-0.2, 0.6)
    ax[2].set_xlim(-1, 5)
    ax[3].set_xlim(-0.2, 0.6)
    ax[4].set_xlim(0, 1.3)
    ax[1].set_yticks(-0.75:0.25:0)
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[4].set_yticks([])
    for a ∈ ax[1:3]
        a.set_ylim(-0.75, 0)
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", linewidth=0.5, linestyle="-")
    end
    colors = pl.cm.viridis(range(1, 0, length=length(profiles2D)))
    for i in eachindex(profiles2D)
        α = αs[i]
        label = latexstring(@sprintf("2D \$\\alpha = 1/%d\$", 1/α))
        z, u, v, w, bzs, u_mask, v_mask, w_mask, bz_mask = profiles2D[i]
        z /= α
        w /= α
        bzs *= α
        ax[1].plot(1e2*u[u_mask],      z[u_mask],  c=colors[i, :], label=label)
        ax[2].plot(1e2*v[v_mask],      z[v_mask],  c=colors[i, :], label=label)
        ax[3].plot(1e2*w[w_mask],      z[w_mask],  c=colors[i, :], label=label)
        ax[4].plot(1 .+ bzs[bz_mask],  z[bz_mask], c=colors[i, :], label=label)
    end
    for i in eachindex(profiles3D)
        α = αs[i]
        label = latexstring(@sprintf("3D \$\\alpha = 1/%d\$", 1/α))
        z, u, v, w, bzs, u_mask, v_mask, w_mask, bz_mask = profiles3D[i]
        z /= α
        w /= α
        bzs *= α
        ax[1].plot(1e2*u[u_mask],      z[u_mask],  "k--", lw=0.5, label=label)
        ax[2].plot(1e2*v[v_mask],      z[v_mask],  "k--", lw=0.5, label=label)
        ax[3].plot(1e2*w[w_mask],      z[w_mask],  "k--", lw=0.5, label=label)
        ax[4].plot(1 .+ bzs[bz_mask],  z[bz_mask], "k--", lw=0.5, label=label)
    end
    ax[2].legend(loc=(-0.8, 0.6))
    savefig("images/aspect_ratios.png")
    println("images/aspect_ratios.png")
    savefig("images/aspect_ratios.pdf")
    println("images/aspect_ratios.pdf")
    plt.close()
end

################################################################################

# mesh()
# convergence()
# zonal_sections()
# vertical_sections()
aspect_ratios()