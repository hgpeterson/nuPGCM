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
B = -∂R( 1/[√2(1 - R²)²] ∫ z ∂R(b) dz )
  = -∂R( 1/[√2(1 - R²)²] ∫ ∂R(zb) dz )
  = -∂R( 1/[√2(1 - R²)²] [∂R(∫ zb dz) + H b(-H) ∂R(H)] )
  = -∂R( 1/[√2(1 - R²)²] ) [∂R(∫ zb dz) + H b(-H) ∂R(H)] - 1/[√2(1 - R²)²] ∂R[∂R(∫ zb dz) + H b(-H) ∂R(H)]
  = -4R/[√2(1 - R²)³] [∂R(∫ zb dz) + H b(-H) ∂R(H)] - 1/[√2(1 - R²)²] ∂R[∂R(∫ zb dz) + H b(-H) ∂R(H)]

√2(1 - R²)³ B = -4R [∂R(∫ zb dz) + H b(-H) ∂R(H)] - (1 - R²) ∂R[∂R(∫ zb dz) + H b(-H) ∂R(H)]
              = -4R [-∂R(γ) + H b(-H) ∂R(H)] - (1 - R²) ∂R[-∂R(γ) + H b(-H) ∂R(H)]
              = -4R f - (1 - R²) ∂R[f]
"""
function compute_barotropic_ode_rhs(R, z, b)
    # γ = -∫ zb dz
    γ = [-trapz(z[i, :].*b[i, :], z[i, :]) for i in eachindex(R)]
    
    # f = -∂R(γ) + H b(-H) ∂R(H)
    H = -z[:, 1]
    f = -differentiate(γ, R) .+ H.*b[:, 1].*differentiate(H, R)

    # rhs = -4R f - (1 - R²) ∂R[f]
    return -4*R .* f .- (1 .- R.^2) .* differentiate(f, R)
end

"""
    G = solve_barotropic_ode(R, f)

Solve the barotropic ODE

    (1 - R²) G'' + 4 G' / R = f

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

        # (1 - R²) G''
        push!(A, (i, i-1, (1 - R[i]^2)*fd_RR[1]))
        push!(A, (i, i,   (1 - R[i]^2)*fd_RR[2]))
        push!(A, (i, i+1, (1 - R[i]^2)*fd_RR[3]))

        # 4 G'/ R 
        push!(A, (i, i-1, 4/R[i]*fd_R[1]))
        push!(A, (i, i,   4/R[i]*fd_R[2]))
        push!(A, (i, i+1, 4/R[i]*fd_R[3]))
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

function main()
    # grid in R
    nR = 2^10
    # R = range(0, 1, length=nR)
    R = chebyshev_nodes(nR) .+ 1

    # depth function
    H = @. 1 - R^2

    # vertical grids
    nz = 2^10
    z = H*chebyshev_nodes(nz)'

    # turbulent mixing coefficient
    κ = [1e-2 + exp(-(z[i, j] + H[i])/0.1) for i in 1:nR, j in 1:nz]

    # parameters
    ε = 1e-2
    μϱ = 1e-4
    γ = θ = 0
    horiz_diff = false
    T = 3e-3*μϱ/ε^2
    Δt = 1e-4*μϱ/ε^2
    params = (μϱ=μϱ, γ=γ, θ=θ, ε=ε, Δt=Δt, T=T, horiz_diff=horiz_diff)

    # solve diffusion problem for each column (except the last one where H = 0)
    b = zeros(nR, nz)
    @showprogress for i in 1:nR-1
        b[i, :] .= diffuse_column(z[i, :], κ[i, :], params)
    end

    # RHS forcing to barotropic ODE
    barotropic_rhs = compute_barotropic_ode_rhs(R, z, b)

    # solve barotropic ODE
    G = solve_barotropic_ode(R, barotropic_rhs)

    return R, z, b, barotropic_rhs, G
end

R, z, b, barotropic_rhs, G = main()

# plot column
i = argmin(abs.(R .- 0.5))
plot_bz(b[i, :], z[i, :]; t=3e-3, filename="images/bz.png")

# plot bowl
fig, ax = plt.subplots(1)
R2 = repeat(R, 1, size(z, 2))
vmax = maximum(abs.(b))
ax.pcolormesh(R2, z, b, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
ax.contour(R2, z, z .+ b, levels=-0.95:0.05:-0.05, colors="k", linewidths=0.3, linestyles="-", alpha=0.5)
ax.spines["left"].set_visible(false)
ax.spines["bottom"].set_visible(false)
ax.set_xlabel(L"R")
ax.set_ylabel(L"z")
ax.set_xlim(0, 1)
savefig("images/b.png")
@info "Saved 'images/b.png'"
plt.close()

# plot B and G
fig, ax = plt.subplots(1)
ax.set_xlim(0, 1)
# ax.set_ylim(-0.2, 0.2)
ax.spines["bottom"].set_visible(false)
ax.axhline(0, color="k", linewidth=0.5)
# ax.set_title(L"\sqrt{2} (1 - R^2)^2 \mathcal{B}")
# ax.plot(R, barotropic_rhs, label=L"\sqrt{2} (1 - R^2)^3 \mathcal{B}")
ax.plot(R, G, label=L"G")
# ax.legend()
ax.set_xlabel(L"R")
savefig("images/barotropic_ode.png")
@info "Saved 'images/barotropic_ode.png'"
plt.close()