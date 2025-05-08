using nuPGCM
using Gridap
using GridapGmsh
using Printf
using PyPlot
using JLD2

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

include("../plots/derivatives.jl")

# simulation folder
set_out_dir!("../sims/sim051")

################################################################################

function save_gridded_data(model, statefile; n=2^6, outfile="gridded.jld2")
    # unpack
    α = model.params.α

    # load state file
    set_state_from_file!(model, statefile)
    u = model.state.u
    v = model.state.v
    w = model.state.w
    b = model.state.b
    t = model.state.t

    # depth function
    H(x) = α*(1 - x[1]^2 - x[2]^2)

    # horizontal grid
    if mod(n, 2) == 0 
        @warn "n must be odd, incrementing to $n + 1"
        n += 1
    end
    x = range(-1, 1, length=n)
    y = range(-1, 1, length=n)
    H = [H([x[i], y[j]]) for i ∈ 1:n, j ∈ 1:n]

    # vertical sigma grid
    σ = range(-1, 0, length=n)

    # points
    points = [Point(x[i], y[j], σ[k]*H[i, j]) for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n][:]

    # evaluate fields
    @time "evaluate u" u = reshape(nan_eval(u, points), n, n, n)
    @time "evaluate v" v = reshape(nan_eval(v, points), n, n, n)
    @time "evaluate w" w = reshape(nan_eval(w, points), n, n, n)
    @time "evaluate b" b = reshape(nan_eval(b, points), n, n, n)

    # save data
    jldsave(outfile; x, y, σ, u, v, w, b, H, t)
    @info "Data saved to '$outfile'"
end

function load_gridded_data(file)
    jldopen(file) do file
        x = file["x"]
        y = file["y"]
        σ = file["σ"]
        u = file["u"]
        v = file["v"]
        w = file["w"]
        b = file["b"]
        H = file["H"]
        t = file["t"]
        return x, y, σ, u, v, w, b, H, t
    end
end

# for contour lines
get_levels(vmax) = [-vmax, -3vmax/4, -vmax/2, -vmax/4, vmax/4, vmax/2, 3vmax/4, vmax]

function compute_U_V(u, v, σ, H)
    # integrate
    U = [trapz(u[i, j, :], σ*H[i, j]) for i in axes(u, 1), j in axes(u, 2)]
    V = [trapz(v[i, j, :], σ*H[i, j]) for i in axes(u, 1), j in axes(u, 2)]

    # NaNs outside of domain
    for i in axes(u, 1), j in axes(u, 2)
        if all(isnan.(u[i, j, :]))
            U[i, j] = NaN
            V[i, j] = NaN
        end
    end

    # # debug: number of NaNs per column
    # nan_count = [sum(isnan.(u[i, j, :])) for i in axes(u, 1), j in axes(u, 2)]
    # # imshow(log.(nan_count/size(u, 1)))
    # # colorbar(label="log(NaN fraction)")
    # imshow(nan_count, vmax=10)
    # colorbar()
    # savefig("nan_count.png")
    # @info "Saved 'nan_count.png'"
    # plt.close()

    return U, V
end

# ∂y(Ψ) = -U and Ψ(x, 0) = 0 → Ψ  = -∫ U dy
function compute_Ψ(U, y)
    U_filled = copy(U)
    U_filled[isnan.(U)] .= 0
    Ψ = zeros(size(U_filled))
    for i in axes(U, 1)
        Ψ[i, :] = -cumtrapz(U_filled[i, :], y)
    end
    Ψ[isnan.(U)] .= NaN
    return Ψ
end

function plot_barotropic(x, y, F; fig=nothing, ax=nothing, label=L"F")
    if ax === nothing
        fig, ax = plt.subplots(1)
    end
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks(-1:1:1)
    ax.set_yticks(-1:1:1)
    vmax = nan_max(abs.(F))
    img = ax.pcolormesh(x, y, F', shading="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=label)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    ax.contour(x, y, F', colors="k", linewidths=0.5, linestyles="-", levels=get_levels(vmax))
    return fig, ax
end

function add_title(ax, t)
    ax.set_title(latexstring(@sprintf("t = %s", sci_notation(t))))
end

function save_plot(fname_base; i=0)
    if i == 0
        fname = @sprintf("%s/images/%s.png", out_dir, fname_base)
    else
        fname = @sprintf("%s/images/%s%03d.png", out_dir, fname_base, i)
    end
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_UV_Ψ(x, y, U, V, Ψ; i=0, t=nothing, β=0)
    # plot UV
    fig, ax = plt.subplots(1, 2, figsize=(3.2*2, 2))
    plot_barotropic(x, y, U, fig=fig, ax=ax[1], label=L"U")
    plot_barotropic(x, y, V, fig=fig, ax=ax[2], label=L"V")
    if t !== nothing
        add_title(ax[1], t)
        add_title(ax[2], t)
    end
    subplots_adjust(wspace=0.4)
    save_plot("UV"; i)

    # plot Psi
    fig, ax = plot_barotropic(x, y, Ψ, label=L"\Psi")
    f_over_H = [(1 + β*y[j])/H[i, j]   for i ∈ eachindex(x), j ∈ eachindex(y)]
    f_over_H[isnan.(Ψ)] .= NaN
    levels = 6*(1:4)/4
    ax.contour(x, y, f_over_H', colors=(0.2, 0.5, 0.2), linewidths=0.5, alpha=0.5, linestyles="-", levels=levels)
    save_plot("psi"; i)
end

function compute_γ(b)
    # resolution
    nx = 2^5
    ny = nx
    nz = 2^5

    # 3D grid
    x = range(-1, 1, length=nx)
    y = range(-1, 1, length=ny)
    σ = chebyshev_nodes(nz)

    # points 
    points = [Point(x[i], y[j], H([x[i], y[j]])*σ[k]) for i ∈ 1:nx, j ∈ 1:ny, k ∈ 1:nz][:]

    # evaluate field
    @time "evals" bs = reshape(nan_eval(b, points), nx, ny, nz)

    # -∫ z*b dz
    γ = -[trapz(H([x[i], y[j]])*σ.*bs[i, j, :], H([x[i], y[j]])*σ) for i ∈ 1:nx, j ∈ 1:ny]

    # NaNs outside of domain
    for i ∈ 1:nx, j ∈ 1:ny
        if all(isnan.(bs[i, j, :]))
            γ[i, j] = NaN
        end
    end

    return x, y, γ
end

function compute_JEBAR(x, y, γ)
    nx = length(x)
    ny = length(y)
    γx = zeros(nx, ny)
    γy = zeros(nx, ny)
    for i ∈ 1:nx
        γy[i, :] = differentiate(γ[i, :], y)
    end
    for j ∈ 1:ny
        γx[:, j] = differentiate(γ[:, j], x)
    end
    # -J(1/H, γ) = (Hx*γy - Hy*γx)/H^2
    return [(Hx([x[i], y[j]])*γy[i, j] - Hy([x[i], y[j]])*γx[i, j])/H([x[i], y[j]])^2 for i ∈ 1:nx, j ∈ 1:ny]
end

function compute_BVE_buoyancy_source(n)
    # load data
    i_save = 3
    file = jldopen(@sprintf("%s/data/gridded_sigma_n%04d_i%03d.jld2", out_dir, n, i_save))
    x = file["x"]
    y = file["y"]
    σ = file["σ"]
    # u = file["u"]
    # v = file["v"]
    # w = file["w"]
    b = file["b"]
    H = file["H"]
    t = file["t"]
    close(file)

    # differentiate H
    Hx = zeros(size(H))
    Hy = zeros(size(H))
    for i ∈ 1:n
        Hx[:, i] = differentiate(H[:, i], x)
        Hy[i, :] = differentiate(H[i, :], y)
    end

    # # debug: compare with analytical -2x and -2y
    # Hx_error = [Hx[i, j] - (-2*x[i]) for i ∈ 1:n, j ∈ 1:n]
    # Hy_error = [Hy[i, j] - (-2*y[j]) for i ∈ 1:n, j ∈ 1:n]
    # fig, ax = plot_barotropic(x, y, Hx_error; label=L"\partial_x H")
    # add_title(ax, t)
    # save_plot("Hx_error"; i=i_save)
    # fig, ax = plot_barotropic(x, y, Hy_error; label=L"\partial_y H")
    # add_title(ax, t)
    # save_plot("Hy_error"; i=i_save)

    # differentiate b
    bξ = zeros(n, n, n)
    bη = zeros(n, n, n)
    bσ = zeros(n, n, n)
    for i ∈ 1:n, j ∈ 1:n 
        bξ[:, i, j] = differentiate(b[:, i, j], x)
        bη[i, :, j] = differentiate(b[i, :, j], y)
        bσ[i, j, :] = differentiate(b[i, j, :], σ)
    end
    bx = [bξ[i, j, k] - σ[k]*Hx[i, j]/H[i, j]*bσ[i, j, k] for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n]
    by = [bη[i, j, k] - σ[k]*Hy[i, j]/H[i, j]*bσ[i, j, k] for i ∈ 1:n, j ∈ 1:n, k ∈ 1:n]

    # compute drag coefficient
    ν_bot = 1
    β = 0
    f = [1 + β*y[j] for i ∈ 1:n, j ∈ 1:n]
    q = @. √(2ν_bot/f)
    r = ν_bot*q./H

    # compute terms 1 and 2
    term1 = [r[i, j]/f[i, j]*H[i, j] * trapz(σ.*(bx[i, j, :] - by[i, j, :]), σ) for i ∈ 1:n, j ∈ 1:n]
    term2 = [r[i, j]/f[i, j]*H[i, j] * trapz(σ.*(bx[i, j, :] + by[i, j, :]), σ) for i ∈ 1:n, j ∈ 1:n]

    # plot terms 1 and 2
    fig, ax = plot_barotropic(x, y, term1; label="term 1")
    add_title(ax, t)
    save_plot("term1_"; i=i_save)
    fig, ax = plot_barotropic(x, y, term2; label="term 2")
    add_title(ax, t)
    save_plot("term2_"; i=i_save)

    # compute convergence
    BVE_buoyancy_source = zeros(n, n)
    for i ∈ 1:n
        BVE_buoyancy_source[:, i] .-= differentiate(term1[:, i], x)
        BVE_buoyancy_source[i, :] .-= differentiate(term2[i, :], y)
    end

    # plot BVE buoyancy source
    fig, ax = plot_barotropic(x, y, BVE_buoyancy_source; label=L"\mathcal{B}")
    add_title(ax, t)
    save_plot("BVE_buoyancy_source"; i=i_save)

    # plot BVE buoyancy source just along y=0
    fig, ax = plt.subplots(1)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"\mathcal{B}")
    # ax.plot(x, BVE_buoyancy_source[:, n÷2])
    ax.plot(x, BVE_buoyancy_source[:, n÷2] .* (sqrt(2)*(1 .- x.^2).^2))
    println(y[n÷2])
    savefig(@sprintf("%s/images/BVE_buoyancy_source_y0_%03d.png", out_dir, i_save))
    println(@sprintf("%s/images/BVE_buoyancy_source_y0_%03d.png", out_dir, i_save))
    plt.close()

    return x, BVE_buoyancy_source
end

# setup model (TODO: load this from file)
ε = 2e-2
α = 1/2
μϱ = 1e0
N² = 1e0/α
Δt = 1e-1
params = Parameters(ε, α, μϱ, N², Δt)
dim = 3
h = 1e-2
mesh = Mesh(@sprintf("../meshes/bowl%sD_%e_%e.msh", dim, h, α))
state = rest_state(mesh)
model = Model(CPU(), params, mesh, nothing, nothing, state) # model with an empty inversion and evolution

# save gridded data
n = 2^8 + 1
save_gridded_data(model, @sprintf("%s/data/state_%016d.jld2", out_dir, 250); n, outfile=@sprintf("%s/data/gridded_n%04d_t%016d.jld2", out_dir, n, 250))
save_gridded_data(model, @sprintf("%s/data/state_%016d.jld2", out_dir, 500); n, outfile=@sprintf("%s/data/gridded_n%04d_t%016d.jld2", out_dir, n, 500))