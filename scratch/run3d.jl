using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function setup()
    ε² = 1e-4
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 1.
    β = 0.
    H(x) = 1 - x[1]^2 - x[2]^2
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    ν(σ, H) = μ*κ(σ, H)
    g_sfc1 = Grid(Triangle(order=1), "meshes/circle/mesh3.h5")
    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H, τx, τy, ν, κ, g_sfc1, nσ=2^6, chebyshev=true, advection=false)
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
    plot_profiles(m, s,  0.5, 0.0, fname="$out_folder/profiles_x=+0.5_y=0.0.png")
    plot_profiles(m, s, -0.5, 0.0, fname="$out_folder/profiles_x=-0.5_y=0.0.png")
    plot_xslices(m, s, 0.0, fname="$out_folder/xslices_y=0.0.png")
    plot_yslices(m, s, 0.0, fname="$out_folder/yslices_x=0.0.png")
    return s
end

m = setup()
s = run(m)

println("Done.")