using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    H(x) = 1 - x[1]^2 - x[2]^2
    Hx(x) = -2x[1]
    Hy(x) = -2x[2]

    # m = ModelSetup3D()

    δ = 0.1
    b = [FEField(x -> x[3] + δ*exp(-(x[3] + H(x))/δ), g) for g ∈ m.b_cols]
    # γ = FEField(x -> -H(x)^3/3 - δ^2*(δ - H(x) - δ*exp(-H(x)/δ)), m.g_sfc)

    # nuPGCM.get_JEBAR(m, b, showplots=true)

    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)

    return m
end

m = bowl()

println("Done.")