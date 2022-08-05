using nuPGCM
using PyPlot
using Printf
using LinearAlgebra
using SuiteSparse
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function plot_convergence()
   fig, ax = subplots() 
   ax.set_title(L"Flat bottom, Bump function at $x = 0$")
   ax.set_xlabel("Resolution (km)")
   ax.set_ylabel("Maximum absolute error (Sv)")
   ax.loglog([79, 53, 26], [2.8e-3, 1.5e-3, 2.3e-4], "o", label="Linear")
   ax.loglog([66, 40, 26], [7.0e-4, 9.7e-5, 1.0e-4], "o", label="Quadratic")
   ax.loglog([60, 40], [9e-4, (40/60)^2*9e-4], "k--")
   ax.loglog([60, 40], [7e-4, (40/60)^4*7e-4], "k--")
   ax.annotate(L"$h^2$", (50, 8e-4))
   ax.annotate(L"$h^4$", (50, 2e-4))
   ax.set_xlim([20, 90])
   ax.set_ylim([5e-5, 5e-3])
   ax.legend(frameon=true, fancybox=false)
   savefig("images/convergence.png")
   println("images/convergence.png")
end

function baroclinic_convergence_1D()
    # params
    ρ₀ = 1000.
    nσ = 2^8
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  
    ν = 1e-3*ones(nσ)
    f = 1e-4
    H = 1e0

    # numerical solution
    baroclinic_LHS = get_baroclinic_LHS(ρ₀, ν, f, H, σ)
    baroclinic_RHS = get_baroclinic_RHS(zeros(nσ), zeros(nσ), 0, 0, 1, 0)
    sol = baroclinic_LHS\baroclinic_RHS
    imap = reshape(1:2*nσ, (2, nσ)) 
    τξ = sol[imap[1, :]]
    τη = sol[imap[2, :]]

    # analytical solution (assuming b = τ₀ = 0)
    q = sqrt(f/2/ν[1])
    Hq = H*q
    z = σ*H
    denom = 1 + exp(-4*Hq) - 2Hq + 2*exp(-4*Hq)*Hq + 2*Hq^2 + 2*exp(-4*Hq)*Hq^2 + 2*exp(-2*Hq)*(2*Hq^2 - 1)cos(2*Hq) - 4*exp(-2*Hq)*Hq*sin(2*Hq)
    c1 = -2*q^2*ν[1]*ρ₀*((exp(-3*Hq) + exp(-Hq))*Hq*cos(Hq) - ((1 + Hq)*exp(-3*Hq) - exp(-Hq)*(Hq - 1))*sin(Hq)) / denom
    c2 =  2*q^2*ν[1]*ρ₀*(((1 + Hq)*exp(-3*Hq) + exp(-Hq)*(Hq - 1))*cos(Hq) - (exp(-Hq) - exp(-3*Hq))*Hq*sin(Hq)) / denom
    c3 =  2*q^2*ν[1]*ρ₀*(Hq + exp(-2*Hq)*Hq*cos(2*Hq) - exp(-2*Hq)*(1 + Hq)*sin(2*Hq)) / denom
    c4 =  2*q^2*ν[1]*ρ₀*((Hq - 1) + exp(-2*Hq)*(1 + Hq)*cos(2*Hq) + exp(-2*Hq)*Hq*sin(2*Hq)) / denom
    
    # add to array
    τξ_a = @. exp(q*z)*(c1*cos(q*z) + c2*sin(q*z)) + exp(-q*(z + H))*(c3*cos(q*(z + H)) + c4*sin(q*(z + H)))
    τη_a = @. exp(q*z)*(c1*sin(q*z) - c2*cos(q*z)) + exp(-q*(z + H))*(c4*cos(q*(z + H)) - c3*sin(q*(z + H)))

    # compare 
    abs_err = abs.(τξ - τξ_a)
    println(@sprintf("Max Abs Error: %1.1e kg m-3 s-1 (i = %d / %d)", maximum(abs_err), argmax(abs_err), nσ))
    println(@sprintf("Max τ:         %1.1e kg m-3 s-1", maximum(abs.(τξ_a))))

    # plot
    fig, ax = subplots()
    ax.set_xlabel(L"Stress $\tau H^2$ (kg m$^{-1}$ s$^{-1}$)")
    ax.set_ylabel(L"Vertical coordinate $z$ (m)")
    ax.plot(τξ*H^2,   z, label=L"$\tau^\xi$")
    ax.plot(τη*H^2,   z, label=L"$\tau^\eta$")
    ax.plot(τξ_a*H^2, z, "k--", lw=0.5, label="Analytical")
    ax.plot(τη_a*H^2, z, "k--", lw=0.5)
    ax.legend()
    # ax.set_ylim([-1, -0.9])
    savefig("images/tau_error.png")
    println("images/tau_error.png")
end

function baroclinic_convergence_full()
    # ref density
    ρ₀ = 1000.

    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)

    # vertical coordinate
    nσ = 2^8
    σ = @. -(cos(π*(0:nσ-1)/(nσ-1)) + 1)/2  

    # coriolis parameter f = f₀ + βη
    f₀ = 1e-4
    β = 0.

    # viscosity
    ν = 1e-3*ones(np, nσ)

    # baroclinic LHS matrices
    baroclinic_LHSs = Array{SuiteSparse.UMFPACK.UmfpackLU}(undef, np) 
    @showprogress "Calculating baroclinic LHSs..." for i=1:np 
        baroclinic_LHSs[i] = get_baroclinic_LHS(ρ₀, ν[i, :], f₀ + β*η[i], H[i], σ)
    end  

    # buoyancy field
    b = zeros(np, nσ)
    for j=1:nσ
        b[:, j] .= 1e-6*H*σ[j] + 0.1*1e-6*H*exp(-(σ[j] + 1)/0.1)
    end

    # buoyancy gradients
    M = nuPGCM.get_M(p, t, C₀)
    M_LU = lu(M)
    Cξ, Cη = nuPGCM.get_Cξ_Cη(p, t, C₀)
    b_x = M_LU\(Cξ*b)
    b_y = M_LU\(Cη*b)
    for i=1:np
        b_x[i, :] += -σ*Hx[i].*differentiate(b[i, :], σ)/H[i] 
        b_y[i, :] += -σ*Hy[i].*differentiate(b[i, :], σ)/H[i]
    end

    # stress due to buoyancy gradients
    baroclinic_RHSs_b = zeros(np, 2*nσ)
    for i=1:np
        coeff = ρ₀*ν[i, :]./(f₀ .+ β*η[i])
        baroclinic_RHSs_b[i, :] = get_baroclinic_RHS(coeff.*b_x[i, :], coeff.*b_y[i, :], 0, 0, 0, 0)
    end
    τξ_b, τη_b = get_τξ_τη(baroclinic_LHSs, baroclinic_RHSs_b)

    # plot
    plot_horizontal(p, t, τξ_b[:, 1]; clabel=L"Buoyancy bottom stress $\tau^\xi_b$ (kg m$^{-1}$ s$^{-2}$)")
    savefig("images/tau_xi_b_pointwise.png")
    println("images/tau_xi_b_pointwise.png")
    plt.close()
    plot_horizontal(p, t, τη_b[:, 1]; clabel=L"Buoyancy bottom stress $\tau^\eta_b$ (kg m$^{-1}$ s$^{-2}$)")
    savefig("images/tau_eta_b_pointwise.png")
    println("images/tau_eta_b_pointwise.png")
    plt.close()
end

function derivative_convergence()
    # basin geo
    p, t, e, np, Lx, Ly, ξ, η, H, Hx, Hy = get_basin_geometry()

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)

    # function and its derivative
    f = ξ.^2 + 3*Lx*ξ
    fξ = 2*ξ .+ 3*Lx

    # approx derivative
    Cξ, Cη = nuPGCM.get_Cξ_Cη(p, t, C₀)
    M = lu(nuPGCM.get_M(p, t, C₀))
    b = Cξ*f
    # M = nuPGCM.get_M_dirichlet(p, t, e, C₀)
    # b = Cξ*f
    # b[e] .= fξ[e]
    fξ0 = M\b

    plot_horizontal(p, t, fξ; contours=false)
    savefig("images/fx.png")
    println("images/fx.png")
    plt.close()
    plot_horizontal(p, t, fξ0; contours=false)
    savefig("images/fx0.png")
    println("images/fx0.png")
    plt.close()
    plot_horizontal(p, t, fξ - fξ0; contours=false)
    savefig("images/abs_err.png")
    println("images/abs_err.png")
    plt.close()
    plot_horizontal(p, t, (fξ - fξ0)./fξ; contours=false)
    savefig("images/rel_err.png")
    println("images/rel_err.png")
    plt.close()

    # absolute error
    abs_err = abs.(fξ - fξ0)
    rel_err = abs_err./abs.(fξ)
    println(@sprintf("%d km", Lx/sqrt(np)/1e3))
    println(@sprintf("Max Abs. Err.: %1.1e", maximum(abs_err)))
    println(@sprintf("Max Rel. Err.: %1.1e", maximum(rel_err)))
    println(@sprintf("Max fξ:        %1.1e", maximum(fξ)))

    # no b.c.
    # 131: 1.6e5
    #  79: 9.6e4
    #  53: 6.2e4
    #  26: 3.2e4

    # dirichlet
    # 131: 2.4e4
    #  79: 2.5e4
    #  53: 1.1e4
    #  26: 7.7e3
end

derivative_convergence()

# baroclinic_convergence_1D()
# baroclinic_convergence_full()

# plot_convergence()

println("Done.")