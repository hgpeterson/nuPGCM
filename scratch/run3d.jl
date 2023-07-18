using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    ε² = 1e-4
    μ = 1e0
    ϱ = 1e-4
    Δt = 1e-3*μ*ϱ/ε²
    f = 0.
    β = 1.
    H(x) = 1 - x[1]^2 - x[2]^2
    τx(x) = 0.
    τy(x) = 0.

    g_sfc1 = Grid(1, "meshes/circle/mesh3.h5")

    m = ModelSetup3D(ε², μ, ϱ, Δt, f, β, H::Function, τx::Function, τy::Function, g_sfc1)

    # m = ModelSetup3D()
    # # b = [FEField(x -> x[3], g) for g ∈ m.b_cols]
    # b = [FEField(x -> H(x)*x[3]^2 + 2/3*x[3]^3, g) for g ∈ m.b_cols]
    # # δ = 0.1
    # # b = [FEField(x -> x[3] + δ*exp(-(x[3] + H(x))/δ), g) for g ∈ m.b_cols]
    # ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    # s = ModelState3D(b, ωx, ωy, χx, χy, 0)
    # # evolve!(m, s)
    # return m, s
end

# m, s = bowl()

println("Done.")