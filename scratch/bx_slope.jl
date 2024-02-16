using nuPGCM
using Printf
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function main()
    # parameters
    Δt = 1e-4
    T = 1e-2
    f = 1.
    L = 1.
    nξ = 2^10 
    nσ = 2^8
    coords = "cartesian"
    periodic = false
    bl = false

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L)
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: bowl
    no_net_transport = true
    H_func(x) = 1 - 0.5x
    Hx_func(x) = -0.5

    # funcs
    κ_func(ξ, σ) = 1e-2 + 10*exp(-H_func(ξ)*(σ + 1)/0.05)
    ν_func(ξ, σ) = 1e-4*κ_func(ξ, σ)
    N2_func(ξ, σ) = 1
    
    # create model struct
    m = ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # set initial state
    b = copy(m.z)
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2D(b, χ, uξ, uη, uσ, i)

    # solve
    evolve!(m, s, T, T, T) 

    # b and bx
    # b = @. m.z + 0.1*exp(-(m.z+m.H)/0.1)
    b = s.b
    bx = ∂x(m, b)
    fig, ax = plt.subplots(1)
    vmax = maximum(abs.(bx))
    vmin = -vmax
    img = ax.pcolormesh(m.x[2:end, :], m.z[2:end, :], bx[2:end, :], cmap="RdBu_r", vmin=vmin, vmax=vmax, rasterized=true, shading="auto")
    levels = -0.8:0.1:-0.1
    ax.contour(m.x, m.z, b, levels=levels, colors="k", alpha=1.0, linestyles="-", linewidths=0.5)
    ax.fill_between(m.x[:, 1], m.z[:, 1], minimum(m.z), color="k", alpha=0.15, lw=0.0)
    ax.plot(m.x[:, 1], m.z[:, 1], "k-", lw=0.5)
    ax.text(0.08, -1.1, s=L"$x$")
    ax.text(-0.05, -0.9, s=L"$z$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([0, m.ξ[end]])
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_title(L"$\partial_x b$")
    savefig("$out_folder/images/bx.png")
    println("$out_folder/images/bx.png")
    plt.close()
end

main()

println("Done.")