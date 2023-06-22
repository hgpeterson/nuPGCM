using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    # m = ModelSetup3D()
    b = [FEField(x -> x[3], g) for g ∈ m.b_cols]
    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)
    s = ModelState3D(b, ωx, ωy, χx, χy, 0)
    evolve!(m, s)
    return m, s
end

m, s = bowl()

println("Done.")