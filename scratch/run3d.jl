using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")

# depth
H(x) = 1 - x[1]^2 - x[2]^2

function setup()
    # params
    μ = 1e4
    ε² = 1e-3
    ϱ = 1e-4
    # Δt = 1e-4*μ*ϱ/ε²
    Δt = 1e-2
    f = 1.
    β = 0.
    # β = 1.
    params = Params(; ε², μ, ϱ, Δt, f, β)

    # geometry
    geom = Geometry(:circle, H, res=3, nσ=0, chebyshev=true)

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

function run3d(m::ModelSetup3D)
    b = FEField(x -> H(x)*x[3], m.geom.g2)
    ωx = DGField(0, m.geom.g1)
    ωy = DGField(0, m.geom.g1)
    χx = DGField(0, m.geom.g1)
    χy = DGField(0, m.geom.g1)
    Ψ = FEField(0, m.geom.g_sfc1)
    s = ModelState3D(b, ωx, ωy, χx, χy, Ψ, [0])

    # t_final = 5e-2*m.params.μ*m.params.ϱ/m.params.ε²
    # t_plot = t_final/5
    # t_save = t_final/50
    t_final = 10
    t_plot = 1
    t_save = 0.1
    evolve!(m, s, t_final, t_plot, t_save)
    return s
end

function postprocess()
    i = 1
    while isfile("$out_folder/state$i.h5")
        s = load_state_3D(m, "$out_folder/state$i.h5")
        title = latexstring(L"$t = $", @sprintf("%.3f", m.params.Δt*s.i[1]))
        filename = @sprintf("%s/psi%03d.png", out_folder, i)
        nuPGCM.quick_plot(s.Ψ,  cb_label=L"Barotropic streamfunction $\Psi$", title=title, filename=filename)
        nuPGCM.plot_u(m, s, 0, i=i)
        i += 1
    end
    # run(`bash -c "make_movie 20 psi"`)
end

m = setup()
# m = load_setup_3D("$out_folder/setup.h5")
s = run3d(m)
postprocess()

println("Done.")