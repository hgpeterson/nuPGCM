using nuPGCM
using PyPlot
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

if !isdir("../output")
    mkdir("../output")
end
set_out_folder("../output")
if !isdir("$out_folder/data")
    mkdir("$out_folder/data")
end
if !isdir("$out_folder/images")
    mkdir("$out_folder/images")
end

# depth
H(x) = 1 - x[1]^2 - x[2]^2

function setup()
    # params
    ε² = 1e-4
    μϱ = 1e-4
    f = 1.
    β = 0.
    params = Params(; ε², μϱ, f, β)

    # geometry
    geom = Geometry(:circle, H, res=3, nσ=0, chebyshev=true)

    # forcing
    τx(x) = 0.
    τy(x) = 0.
    κ(σ, H) = 1e-2 + exp(-H*(σ + 1)/0.1)
    ν(σ, H) = κ(σ, H)
    forcing = Forcing(geom, τx, τy, ν, κ)

    # setup and save
    m = ModelSetup3D(params, geom, forcing, advection=true)
    save_setup(m)

    return m
end

function run3d(m::ModelSetup3D)
    b = FEField(x -> H(x)*x[3], m.geom.g2)
    s = initial_state(m, b)

    Δt = 1e-4
    t_save = 1e-3
    t_final = 1e-1
    evolve!(m, s, t_final, t_save; Δt)
    return s
end

m = setup()
s = run3d(m)

println("Done.")
