using nuPGCM
using PyPlot

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/")

function emulate_2D(; bl = false)
    # parameters (see `setup.jl`)
    f = 8.753044701640954e-5 
    L = 5e6
    nξ = 2^8 
    nσ = 2^8
    coords = "axisymmetric"
    periodic = false

    # grids: even spacing in ξ and chebyshev in σ (unless bl)
    ξ = collect(L/nξ:L/nξ:L)
    if bl
        σ = collect(-1:1/(nσ-1):0)
    else
        σ = @. -(cos(pi*(0:nσ-1)/(nσ-1)) + 1)/2  
    end
    
    # topography: sine
    no_net_transport = true
    # H₀ = 4e3
    # H₀ = 2e3
    H₀ = 2e2
    Δ = L/5 # width of gaussian for bathtub
    G(x) = 1 - exp(-x^2/(2*Δ^2)) # gaussian for bathtub
    Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
    heaviside(x) = (1 + sign(x))/2
    G_bump(x) = -exp(-16*Δ^2/(16*Δ^2 - x^2))*heaviside(4*Δ - x)
    Gx_bump(x) = 32*x*Δ^2*G_bump(x)/(16*Δ^2 - x^2)^2
    # H_func(x) = H₀*G(x - L) + 5
    # Hx_func(x) = H₀*Gx(x - L)
    # H_func(x) = H₀*G(x - 0) + 2e3
    # Hx_func(x) = H₀*Gx(x - 0)
    H_func(x) = H₀*G_bump(x - 0) + 2e3
    Hx_func(x) = H₀*Gx_bump(x - 0)
    # H_func(x) = H₀ + 0*x
    # Hx_func(x) = 0*x

    # plot(ξ, -H_func.(ξ))
    # savefig("debug.png")
    # plt.close()
    # error()

    # diffusivity
    # κ0 = 6e-5
    # κ1 = 2e-3
    # h = 200
    κ0 = 1e-1
    κ1 = 0
    h = 1
    κ_func(ξ, σ) = κ0 + κ1*exp(-H_func(ξ)*(σ + 1)/h)

    # viscosity
    μ = 1e0
    ν_func(ξ, σ) = μ*κ_func(ξ, σ)

    # stratification
    N2 = 1e-6
    N2_func(ξ, σ) = N2
    # δ = 1000 # decay scale (m)
    # N2 = 1e-6*exp(H_func(L/4)/δ) # match bottom strat with const N2 at center of ridge flank
    # N2_func(ξ, σ) = N2*exp(H_func(ξ)*σ/δ)
    
    # timestepping
    Δt = 0.
    
    # create model struct
    m = ModelSetup2DPG(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, H_func, Hx_func, ν_func, κ_func, N2_func, Δt)

    # set initial state
    b = zeros(nξ, nσ)
    for j=1:nσ
        b[:, j] .= m.N2[:, j].*m.H*m.σ[j] + 0.1*m.N2[:, j].*m.H*exp(-(m.σ[j] + 1)/0.1)
    end
    # for i=1:nξ
    #     # b[i, :] .= m.N2[i, :]*m.H[i].*m.σ * (1 - 0.1*exp(-(m.ξ[i] - m.L/2)^2/2/(m.L/8)^2))
    #     # if m.L/4 < m.ξ[i] < 3*m.L/4
    #     #     b[i, :] .= m.N2[i, :]*m.H[i].*m.σ * (1 - 0.1*exp(-m.L^2/16/(m.L^2/16 - (m.ξ[i] - m.L/2)^2)))
    #     # else
    #     #     b[i, :] .= m.N2[i, :]*m.H[i].*m.σ
    #     # end
    #     Δ = 0.9*m.L
    #     if m.ξ[i] < Δ
    #         b[i, :] .= m.N2[i, :]*m.H[i].*m.σ * (1 - 0.1*exp(-Δ^2/(Δ^2 - m.ξ[i]^2)))
    #     else
    #         b[i, :] .= m.N2[i, :]*m.H[i].*m.σ
    #     end
    # end
    χ, uξ, uη, uσ, U = invert(m, b)
    i = [1]
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)

    return m, s
end

m2D, s2D = emulate_2D()
save_setup(m2D, "setup2D.h5")
save_state(s2D, "state2D.h5")

ridge_plot(m2D, s2D, s2D.uξ, "", L"Zonal velocity $u^x$ (m s$^{-1}$)"; style="pcolormesh")
savefig("images/ux2D.png")
println("images/ux2D.png")
plt.close()

ridge_plot(m2D, s2D, s2D.uη, "", L"Meridional velocity $u^y$ (m s$^{-1}$)"; style="pcolormesh")
savefig("images/uy2D.png")
println("images/uy2D.png")
plt.close()

# # load 2D
# m2D_hr = load_setup_2D("../output/setup2D.h5")
# s2D_hr = load_state_2D("../output/state2D.h5")

# # comparison points
# using Dierckx
# ξ₀s = 0.5e6:0.5e6:4.5e6
# for i=1:size(ξ₀s, 1)
#     ξ₀ = ξ₀s[i]

#     # interps
#     H = Spline1D(m2D.ξ, m2D.H)(ξ₀)
#     uξ = Spline2D(m2D.ξ, m2D.σ, s2D.uξ)
#     uη = Spline2D(m2D.ξ, m2D.σ, s2D.uη)
#     uξ_hr = Spline2D(m2D_hr.ξ, m2D_hr.σ, s2D_hr.uξ)
#     uη_hr = Spline2D(m2D_hr.ξ, m2D_hr.σ, s2D_hr.uη)

#     # plot
#     fig, ax = subplots(1, 2, figsize=(2*1.955, 3.176))
#     ax[1].set_title(latexstring(L"Comparison point: $x = $", @sprintf("%d", ξ₀/1e3), " km")) 
#     ax[1].set_xlabel(L"Zonal velocity $u^x$ ($\times$ 10$^{-3}$ m s$^{-1}$)")
#     ax[2].set_xlabel(L"Meridional velocity $u^y$ ($\times$ 10$^{-3}$ m s$^{-1}$)")
#     ax[1].set_ylabel(L"Vertical coordinate $z$ (km)")
#     ax[1].plot(1e3*uξ_hr.(ξ₀, m2D_hr.σ),  H*m2D_hr.σ/1e3, label="2D HR")
#     ax[1].plot(1e3*uξ.(ξ₀, m2D.σ),  H*m2D.σ/1e3, label="2D", "--")
#     ax[2].plot(1e3*uη_hr.(ξ₀, m2D_hr.σ),  H*m2D_hr.σ/1e3, label="2D HR")
#     ax[2].plot(1e3*uη.(ξ₀, m2D.σ),  H*m2D.σ/1e3, label="2D", "--")
#     ax[1].legend()
#     ax[1].set_ylim([-H/1e3, (-H + 100)/1e3])
#     savefig("images/ux_uy_column$(i)_2D.png")
#     println("images/ux_uy_column$(i)_2D.png")
#     plt.close()
# end