using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function bowl()
    # H(x) = 1 - x[1]^2 - x[2]^2
    # Hx(x) = -2x[1]
    # Hy(x) = -2x[2]

    m = ModelSetup3D()

    b = [FEField(x -> x[3], g) for g ∈ m.b_cols]

    nuPGCM.get_JEBAR(m, b, showplots=true)

    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)
    s = ModelState3D(b, ωx, ωy, χx, χy, 0)

    s = evolve(m, s)

    return m, s
end

m, s = bowl()

println("Done.")