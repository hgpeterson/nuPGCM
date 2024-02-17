using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function main(; bl=false)
    # parameters
    f = 5.5e-5
    N2 = 1e-6
    nz = 2^8
    H = 2e3
    θ = 2.5e-3         
    transport_constraint = true
    U = [0.0]
    Δt = 10*secs_in_day

    # grid: chebyshev unless bl
    if bl
        z = collect(-H:H/(nz-1):0) # uniform
    else
        z = @. -H*(cos(pi*(0:nz-1)/(nz-1)) + 1)/2 # chebyshev 
    end
    
    # diffusivity
    κ0 = 6e-5
    κ1 = 2e-3
    h = 200
    κ_func(z) = κ0 + κ1*exp(-(z + H)/h)
    κ_z_func(z) = -κ1/h*exp(-(z + H)/h)

    # viscosity
    μ = 1e0
    ν_func(z) = μ*κ_func(z)
    
    # create model struct
    m = ModelSetup1D(bl, f, nz, z, H, θ, ν_func, κ_func, κ_z_func, N2, Δt, transport_constraint, U)

    # invert
    b = zeros(nz)
    χ, u, v = invert(m, b)
    i = [1]
    s = ModelState1D(b, χ, u, v, i)

    # solve transient
    evolve!(m, s, 15*secs_in_year, 15*secs_in_year) 
    v = s.v
    V = trapz(v, z)

    # three steps
    v_TW = v
    v_TW_no_V = @. v_TW - V/H
    q = 0.05
    zB = z .+ H
    v_TW_no_V_BL = @. v_TW_no_V - v_TW_no_V[1]*exp(-q*zB)*(cos(q*zB) + sin(q*zB))

    # plot
    α = 0.4
    ax = plotsetup()
    ax.plot(v_TW, z, "C0")
    savefig("$out_folder/images/uy_TW1.png")
    println("$out_folder/images/uy_TW1.png")
    plt.close()
    ax = plotsetup()
    ax.plot(v_TW, z, "C0", alpha=α)
    ax.plot(v_TW_no_V, z, "C0")
    savefig("$out_folder/images/uy_TW2.png")
    println("$out_folder/images/uy_TW2.png")
    plt.close()
    ax = plotsetup()
    ax.plot(v_TW, z, "C0", alpha=α)
    ax.plot(v_TW_no_V, z, "C0", alpha=α)
    ax.plot(v_TW_no_V_BL, z, "C0")
    savefig("$out_folder/images/uy_TW3.png")
    println("$out_folder/images/uy_TW3.png")
    plt.close()
end

function plotsetup()
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(false)
    ax.axvline(0, lw=0.5, c="k")
    ax.text(-2e-3, 0, s=L"z")
    ax.text(2e-2, -2080, s=L"u^y")
    ax.set_xlim(-2.5e-2, 2.5e-2)
    return ax
end

main()

println("Done.")