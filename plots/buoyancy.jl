using ProgressMeter 
using PyPlot

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

include("derivatives.jl")
include("../scratch/1D.jl")

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

function diffuse_columns()
    # grid in x
    nx = 2^10
    x = range(0, 1, length=nx)

    # depth function
    H = @. 1 - x^2
    Hx = -2*x
    θs = -atan.(Hx)

    # vertical grids
    nz = 2^10
    σ = chebyshev_nodes(nz)
    z = H*σ'

    # turbulent mixing coefficient
    κ = [1e-2 + exp(-(z[i, j] + H[i])/0.1) for i in 1:nx, j in 1:nz]

    # parameters
    ε = 1e-2
    μϱ = 1e-4
    α = 1/2
    horiz_diff = false
    T = 5e-2*μϱ/ε^2
    Δt = 1e-4*μϱ/ε^2

    # solve diffusion problem for each column (except the last one where H = 0)
    b = zeros(nx, nz)
    @showprogress for i in 1:nx-1
        params = (μϱ=μϱ, α=α, θ=θs[i], ε=ε, Δt=Δt, T=T, horiz_diff=horiz_diff)
        b[i, :] .= diffuse_column(z[i, :], κ[i, :], params)
    end

    # compute bx = ∂ξ(b) - σ*Hx/H ∂σ(b)
    bx = zeros(nx, nz)
    for j in 1:nz
        # ∂ξ(b)
        bx[:, j] .+= differentiate(b[:, j], x)
    end
    for i in 1:nx
        # -σ*Hx/H ∂σ(b)
        bx[i, :] .-= σ*Hx[i]/H[i].*differentiate(b[i, :], σ)
    end
    # for H = 0
    bx[nx, :] .= 0

    return x, z, b, bx, T
end

# # compute b
# x, z, b, bx, T = diffuse_columns()

# # plot
# fig, ax = plt.subplots(1)
# vmax = maximum(abs.(bx))
# xx = repeat(x, 1, size(z, 2))
# img = ax.pcolormesh(xx, z, bx, cmap="RdBu_r", shading="auto", rasterized=true, vmin=-vmax, vmax=vmax)
# plt.colorbar(img, ax=ax, label=L"Buoyancy gradient $\partial_x b$")
# ax.contour(xx, z, z .+ b, levels=-0.95:0.05:-0.05, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
# ax.spines["left"].set_visible(false)
# ax.spines["bottom"].set_visible(false)
# ax.set_xlabel(L"x")
# ax.set_ylabel(L"z")
# savefig("bx.png")
# @info "Saved 'bx.png'"
# plt.close()

# save 
jldsave("buoyancy.jld2"; x, z, b, bx, t=T)
@info "Saved 'buoyancy.jld2'"