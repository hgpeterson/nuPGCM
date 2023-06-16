using nuPGCM
using PyPlot

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

# m = ModelSetup3D()

δ = 0.1
H(x) = 1 - x[1]^2 - x[2]^2
Hx(x) = -2x[1]
Hy(x) = -2x[2]
b(x) = x[3] + δ*exp(-(x[3] + H(x))/δ)
γ = FEField(x -> -H(x)^3/3 - δ^2*(δ - H(x) - δ*exp(-H(x)/δ)), m.g_sfc)
# nuPGCM.quick_plot(γ, L"\gamma", "scratch/images/gamma.png")
# JEBAR(x) = ∂y(γ, x)*Hx(x) - ∂x(γ, x)*Hy(x)
# nuPGCM.quick_plot(JEBAR, m.g_sfc, L"-H^2 J(1/H, \gamma)", "scratch/images/JEBAR.png")

ωx, ωy, χx, χy, Ψ = invert(m, b, γ, showplots=true, nonzero_b=true)

println("Done.")