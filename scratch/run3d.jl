using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    m = ModelSetup3D()
    # b = [FEField(x -> x[3], g) for g ∈ m.b_cols]
    H(x) = 1 - x[1]^2 - x[2]^2
    b = [FEField(x -> H(x)*x[3]^2 + 2/3*x[3]^3, g) for g ∈ m.b_cols]
    # δ = 0.1
    # b = [FEField(x -> x[3] + δ*exp(-(x[3] + H(x))/δ), g) for g ∈ m.b_cols]
    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    s = ModelState3D(b, ωx, ωy, χx, χy, 0)
    # evolve!(m, s)
    return m, s
end

m, s = bowl()

println("Done.")