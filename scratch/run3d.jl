using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function setup()
    ε² = 1e-2
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 1.
    β = 0.
    H(x) = 1 - x[1]^2 - x[2]^2
    # τx(x) = -cos(π*x[2])
    τx(x) = 0.
    τy(x) = 0.
    g_sfc1 = Grid(1, "meshes/circle/mesh2.h5")
    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, g_sfc1)
    return m
end

function run(m)
    H(x) = 1 - x[1]^2 - x[2]^2
    # b = FEField(x -> H(x)^3*(x[3]^2 + 2/3*x[3]^3), m.g2)
    b = FEField(x -> H(x)*x[3], m.g2)
    # b = FEField(x -> H(x)*x[3] + 0.1*exp(-H(x)*(x[3] + 1)/0.1), m.g2)
    # b = FEField(x -> exp(-(x[1]^2 + x[2]^2 + (H(x)*x[3] + 0.5)^2)/0.02), m.g2)
    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, 0)
    evolve!(m, s)
    return s
end

m = setup()
s = run(m)

println("Done.")