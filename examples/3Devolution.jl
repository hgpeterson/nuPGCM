using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function setup_circle_bathtub()
    # mesh
    p, t, e = load_mesh("../meshes/circle2.h5")
    np = size(p, 1)

    # width
    Lx = 5e6
    Ly = 5e6
    p[:, 1] *= Lx
    p[:, 2] *= Ly
    ξ = p[:, 1]
    η = p[:, 2]

    # depth
    H₀ = 2e3
    Δ = Lx/5 
    G(r) = 1 - exp(-r^2/(2*Δ^2)) 
    Gr(r) = r/Δ^2*exp(-r^2/(2*Δ^2))
    H  = @. H₀*G(sqrt(ξ^2 + η^2) - Lx) + 0.01
    Hx = @. H₀*Gr(sqrt(ξ^2 + η^2) - Lx)*ξ/sqrt(ξ^2 + η^2)
    Hy = @. H₀*Gr(sqrt(ξ^2 + η^2) - Lx)*η/sqrt(ξ^2 + η^2)

    # use bl theory?
    bl = false

    # ref density
    ρ₀ = 1000.

    # vertical coordinate
    nσ = 2^7
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  

    # coriolis parameter f = f₀ + βη
    f₀ = 1e-4
    β = 0.

    # diffusivity and viscosity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    μ = 1e0
    κ = zeros(np, nσ)
    for i=1:nσ
        κ[:, i] = @. κ0 + κ1*exp(-H*(σ[i] + 1)/h)
    end
    ν = μ*κ

    # stratification
    N² = 1e-6*ones(np, nσ)

    # timestep
    Δt = 10. *secs_in_day

    # model setup struct
    return ModelSetup3DPG(bl, ρ₀, f₀, β, Lx, Ly, p, t, e, σ, H, Hx, Hy, ν, κ, N², Δt)
end

function run_circle_bathtub(m)
    # initial state
    b = zeros(m.np, m.nσ)
    N² = m.N²[1, 1] # constant 
    for j=1:m.nσ
        b[:, j] .= N²*m.H*m.σ[j] 
    end
    ξ_slice = (-m.Lx + 1e4):m.Lx/2^7:(m.Lx - 1e4)
    η₀ = 0
    Ψ = zeros(m.np)
    uξ = zeros(m.np, m.nσ)
    uη = zeros(m.np, m.nσ)
    uσ = zeros(m.np, m.nσ)
    s = ModelState3DPG(b, Ψ, uξ, uη, uσ, [1])

    # ax = plot_ξ_slice(m, s, s.b, ξ_slice, η₀; clabel=L"Buoyancy $b$ (m s$^{-2}$)", contours=false)
    # ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
    # ax.set_ylim([-maximum(m.H)/1e3, 0])
    # savefig("images/b.png")
    # println("images/b.png")
    # plt.close()

    # evolve
    evolve!(m, s, m.Δt, m.Δt)
end

# m = setup_circle_bathtub()
run_circle_bathtub(m)