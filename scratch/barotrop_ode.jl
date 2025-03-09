using ProgressMeter 
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

include("../plots/derivatives.jl")
include("1D.jl")

function diffuse_column(z, κ, params)
    # build matrices
    LHS_b, RHS_b, rhs_b = build_b(z, κ, params)
    LHS_b = lu(LHS_b)

    # initial condition
    b = zeros(length(z))

    # run
    n_steps = Int(round(params.T/params.Δt))
    for i ∈ 1:n_steps
        ldiv!(b, LHS_b, RHS_b*b + rhs_b)
    end

    return b
end

"""
√2 R (1 - R^2)^3 B = R (1 - R^2) γ'' 
                   + (1 + 3 R^2) γ' 
                   + 2 R^2 (1 - R^2)^2 b'(-H)
                   + 4 R (1 - R^2) b(-H)
"""
function compute_barotropic_ode_rhs(R, z, b)
    # γ = -∫ zb dz
    γ = [-trapz(z[i, :].*b[i, :], z[i, :]) for i in eachindex(R)]
    γR = differentiate(γ, R)
    γRR = differentiate(γR, R)
    fig, ax = plt.subplots(1)
    ax.spines["bottom"].set_visible(false)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ymargin(0.1)
    ax.plot(R, γ, label=L"\gamma")
    ax.plot(R, γR, label=L"\gamma'")
    ax.plot(R, γRR, label=L"\gamma''")
    ax.set_xlabel(L"R")
    ax.legend()
    savefig("images/gamma.png")
    @info "Saved 'images/gamma.png'"
    plt.close()

    # bottom buoyancy
    bbot = b[:, 1]
    bbotR = differentiate(bbot, R)
    fig, ax = plt.subplots(1)
    ax.spines["bottom"].set_visible(false)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ymargin(0.1)
    ax.plot(R, bbot, label=L"b|_{-H}")
    ax.plot(R, bbotR, label=L"b'|_{-H}")
    ax.legend()
    ax.set_xlabel(L"R")
    savefig("images/bbot.png")
    @info "Saved 'images/bbot.png'"
    plt.close()

    # compute terms
    rhs1 = R.*(1 .- R.^2).*γRR
    rhs2 = (1 .+ 3*R.^2).*γR
    rhs3 = 2*R.^2 .*(1 .- R.^2).^2 .*differentiate(bbot, R)
    rhs4 = 4*R.*(1 .- R.^2).*bbot

    # plot
    fig, ax = plt.subplots(1)
    ax.spines["bottom"].set_visible(false)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_ymargin(0.1)
    ax.plot(R, rhs1, label="Term 1")
    ax.plot(R, rhs2, label="Term 2")
    ax.plot(R, rhs3, label="Term 3")
    ax.plot(R, rhs4, label="Term 4")
    ax.plot(R, rhs1 + rhs2 + rhs3 + rhs4, "k", label="Total")
    ax.legend()
    ax.set_xlabel(L"R")
    ax.set_ylabel(L"\sqrt{2} R (1 - R^2)^3 \mathcal{B}")
    savefig("images/barotropic_rhs.png")
    @info "Saved 'images/barotropic_rhs.png'"
    plt.close()

    return rhs1 + rhs2 + rhs3 + rhs4
end

"""
    G = solve_barotropic_ode(R, f)

Solve the barotropic ODE

    R (1 - R²) G'' + (1 + 3 R²) G' = f

with boundary conditions 

    G'(0) = 0 and G(1) = 0

using finite differences.
"""
function solve_barotropic_ode(R, f)
    # build FD matrix
    N = length(R)
    A = Tuple{Int64,Int64,Float64}[]
    for i in 2:N-1
        # first and second derivative stencils
        fd_R  = mkfdstencil(R[i-1:i+1], R[i], 1)
        fd_RR = mkfdstencil(R[i-1:i+1], R[i], 2)

        # R (1 - R²) G''
        push!(A, (i, i-1, R[i]*(1 - R[i]^2)*fd_RR[1]))
        push!(A, (i, i,   R[i]*(1 - R[i]^2)*fd_RR[2]))
        push!(A, (i, i+1, R[i]*(1 - R[i]^2)*fd_RR[3]))

        # (1 + 3 R²) G' 
        push!(A, (i, i-1, (1 + 3*R[i]^2)*fd_R[1]))
        push!(A, (i, i,   (1 + 3*R[i]^2)*fd_R[2]))
        push!(A, (i, i+1, (1 + 3*R[i]^2)*fd_R[3]))
    end

    # G'(0) = 0
    fd_R = mkfdstencil(R[1:3], R[1], 1)
    push!(A, (1, 1, fd_R[1]))
    push!(A, (1, 2, fd_R[2]))
    push!(A, (1, 3, fd_R[3]))

    # G(1) = 0
    push!(A, (N, N, 1))

    # sparse matrix from I, J, V tuple
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # set b.c. in RHS
    rhs = zeros(N)
    rhs[2:end-1] = f[2:end-1]

    # solve
    return A\f
end

function diffuse_columns()
    # grid in R
    nR = 2^10
    # R = range(0, 1, length=nR)
    R = chebyshev_nodes(nR) .+ 1

    # depth function
    H = @. 1 - R^2
    HR = -2*R
    θs = -atan.(HR)

    # vertical grids
    nz = 2^10
    z = H*chebyshev_nodes(nz)'

    # turbulent mixing coefficient
    κ = [1e-2 + exp(-(z[i, j] + H[i])/0.1) for i in 1:nR, j in 1:nz]

    # parameters
    ε = 1e-2
    μϱ = 1e-4
    α = 1/2
    horiz_diff = false
    T = 3e-3*μϱ/ε^2
    Δt = 1e-4*μϱ/ε^2

    # solve diffusion problem for each column (except the last one where H = 0)
    b = zeros(nR, nz)
    @showprogress for i in 1:nR-1
        params = (μϱ=μϱ, α=α, θ=θs[i], ε=ε, Δt=Δt, T=T, horiz_diff=horiz_diff)
        b[i, :] .= diffuse_column(z[i, :], κ[i, :], params)
    end

    # add z to b to make it the full buoyancy field
    b .+= z
    
    return R, z, b
end

# R, z, b = diffuse_columns()
barotropic_rhs = compute_barotropic_ode_rhs(R, z, b)
G = solve_barotropic_ode(R, barotropic_rhs)

# # plot column
# i = argmin(abs.(R .- 0.5))
# plot_bz(b[i, :], z[i, :]; t=3e-3, filename="images/bz.png")

# # plot column vs full field
# d = jldopen("../sims/sim048/data/gridded_sigma_beta0.0_n0257_i003.jld2", "r")
# b_full = d["b"]
# x = d["x"]
# y = d["y"]
# σ = d["σ"]
# H = d["H"]
# close(d)
# i = argmin(abs.(x .- 0.5))
# j = length(y) ÷ 2 + 1
# z_full = σ*H[i, j]
# fig, ax = plt.subplots(1, figsize=(2, 3.2))
# ax.set_ylabel(L"Vertical coordinate $z$")
# ax.set_xlabel(L"Stratification $\partial_z b$")
# ax.set_xlim(0, 1.5)
# bz_full = differentiate(b_full[i, j, :], z_full)
# ax.plot(1 .+ bz_full[isnan.(bz_full) .== 0], z_full[isnan.(bz_full) .== 0], label="3D")
# i = argmin(abs.(R .- 0.5))
# bz1D = differentiate(b[i, :], z[i, :])
# ax.plot(bz1D, z[i, :], "k--", lw=0.5, label="1D")
# ax.legend()
# ax.set_title(latexstring(@sprintf("\$t = %s\$", sci_notation(3e-3))))
# savefig("images/bz_full.png")
# @info "Saved 'images/bz_full.png'"
# plt.close()

# # plot bowl
# fig, ax = plt.subplots(1)
# R2 = repeat(R, 1, size(z, 2))
# vmax = maximum(abs.(b))
# img = ax.pcolormesh(R2, z, b, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
# plt.colorbar(img, ax=ax, label=L"Buoyancy $b$")
# ax.contour(R2, z, b, levels=-0.95:0.05:-0.05, colors="k", linewidths=0.3, linestyles="-", alpha=0.5)
# ax.spines["left"].set_visible(false)
# ax.spines["bottom"].set_visible(false)
# ax.set_xlabel(L"R")
# ax.set_ylabel(L"z")
# ax.set_xlim(0, 1)
# savefig("images/b.png")
# @info "Saved 'images/b.png'"
# plt.close()

# plot Ψ
d = jldopen("../sims/sim048/data/psi_beta0.0_n0257_003.jld2", "r")
x = d["x"]
Ψ = d["Ψ"]
close(d)
fig, ax = plt.subplots(1)
ax.set_xlim(0, 1)
# ax.set_ymargin(0.1)
ax.spines["bottom"].set_position("zero")
ax.xaxis.set_label_coords(0.5, 1.25)
ax.tick_params(axis="x", top=true, labeltop=true, bottom=false, labelbottom=false)
ax.axhline(0, color="k", linewidth=0.5)
i = size(Ψ, 2) ÷ 2 + 1
ax.plot(x, 1e2*Ψ[:, i], label="Model")
ax.plot(R, 1e2*G, "k--", lw=0.5, label="Theory")
ax.legend()
ax.set_xticks(0:0.5:1)
ax.set_xlabel(L"Zonal coordinate $x$")
ax.set_ylabel(L"Barotropic streamfunction $\Psi$ ($\times 10^{-2}$)")
savefig("images/barotropic_ode.png")
@info "Saved 'images/barotropic_ode.png'"
plt.close()
