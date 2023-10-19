using nuPGCM
using PyPlot
using Printf
import Base: run

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

# depth
H(x) = 1 - x[1]^2 - x[2]^2
# H(x) = 4*x[1]*(1 - x[1])*(1 - x[2]^2)
# H(x) = 1 + 0*x[1]

function setup()
    # params
    μ = 1e0
    # ε² = 4e-6
    ε² = 1e-2
    # ϱ = 7e-4
    ϱ = 1e0
    # ϱ = 1e-4
    Δt = 1e-4*μ*ϱ/ε²
    # Δt = 1e-4
    f = 1.
    # β = 0.
    β = 1.
    params = Params(; ε², μ, ϱ, Δt, f, β)

    # geometry
    geom = Geometry(:circle, H, res=2, nσ=0, chebyshev=false)

    # forcing
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    # κ(σ, H) = 1 + 0*σ*H
    ν(σ, H) = κ(σ, H)
    forcing = Forcing(geom, τx, τy, ν, κ)

    # setup and save
    m = ModelSetup3D(params, geom, forcing, advection=true)
    save_setup(m)

    return m
end

function run(m::ModelSetup3D)
    b = FEField(x -> H(x)*x[3], m.geom.g2)
    # b = FEField(x -> H(x)*x[3] + 0.1*exp(-H(x)*(x[3] + 1)/0.1), m.geom.g2)
    # b = FEField(x -> exp(-(x[1]^2 + x[2]^2 + (H(x)*x[3] + H([0, 0])/2)^2)/0.02), m.geom.g2)

    ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=false)
    # ωx, ωy, χx, χy, Ψ = invert(m, b, showplots=true)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, 0)
    # s.b.values[:] = FEField(x -> exp(-((x[1] - 0.5)^2 + x[2]^2 + (H(x)*x[3] + 0.75)^2)/0.02), m.g2).values
    # s.b.values[:] = FEField(x -> exp(-((x[1] - 0.8)^2 + x[2]^2 + (H(x)*x[3] + H([0, 0.8]))^2)/0.02), m.g2).values

    # t_final = 5.0
    # t_final = 5e-2
    t_final = 5e-2*m.params.μ*m.params.ϱ/m.params.ε²
    t_plot = t_final/5
    t_save = t_final/50
    evolve!(m, s, t_final, t_plot, t_save)
    return s
end

function postprocess()
    # i = 1
    # while isfile("$out_folder/state$i.h5")
    #     s = load_state_3D(m, "$out_folder/state$i.h5")
    #     nuPGCM.quick_plot(s.Ψ, L"Barotropic streamfunction $\Psi$", @sprintf("%s/psi%03d.png", out_folder, i))
    #     # nuPGCM.plot_u(m, s, 0, i=i)
    #     i += 1
    # end
    run(`bash -c "make_movie 20 psi"`)
end

# m = setup()
# s = run(m)
postprocess()

println("Done.")