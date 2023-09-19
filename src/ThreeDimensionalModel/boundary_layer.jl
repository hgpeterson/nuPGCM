function test_1d()
    # inputs
    ε² = 1e-4
    ε = sqrt(ε²)
    nz = 2^8
    H = 1
    z = @. -H*(cos(π*(0:nz-1)/(nz-1)) + 1)/2
    bx = z
    by = ones(nz)
    Ux = 0
    Uy = 0
    τx = 0
    τy = 0
    y = 1

    # numerical sol

    A = build_baroclinic_LHS(z, ε², y)
    r = get_baroclinic_RHS(z, bx, by, Ux, Uy, τx, τy, ε²)
    sol = A\r
    ωx = sol[1:nz]
    ωy = sol[nz+1:2nz]
    χx = sol[2nz+1:3nz]
    χy = sol[3nz+1:4nz]

    # BL sol
    q = sqrt(y/2)
    z_b = (z .+ H)/ε
    z_s = z/ε

    # # transport
    # c1 = -q/H
    # c2 = +q/H
    # χx_I0 = 0
    # χy_I0 = @. -(z + H)/H
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))
    # χy_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωx_BL = 1/ε*ωx_B1
    # ωy_BL = 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ ε*(χx_I1 .+ χx_B1)
    # χy_BL = χy_I0 .+ ε*(χy_I1 .+ χy_B1)

    # # wind
    # c1 = c2 = -1/(2*H*q)
    # χx_I0 = @. (z + H)/(2*H*q^2)
    # χy_I0 = 0
    # ωx0_B0 = @. -exp(q*z_s)*sin(q*z_s)
    # ωy0_B0 = @. exp(q*z_s)*cos(q*z_s)
    # χx0_B0 = @. -1/(2*q^2)*exp(q*z_s)*cos(q*z_s)
    # χy0_B0 = @. -1/(2*q^2)*exp(q*z_s)*sin(q*z_s)
    # χx_I1 = @. -c2*z/(2*H*q^2)
    # χy_I1 = @. +c1*z/(2*H*q^2)
    # ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    # χx_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))
    # χy_B1 = @. 1/(2*q^2)*exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    # ωx_BL = 1/ε²*ωx0_B0 .+ 1/ε*ωx_B1
    # ωy_BL = 1/ε²*ωy0_B0 .+ 1/ε*ωy_B1
    # χx_BL = χx_I0 .+ χx0_B0 .+ ε*(χx_I1 .+ χx_B1)
    # χy_BL = χy_I0 .+ χy0_B0 .+ ε*(χy_I1 .+ χy_B1)

    # buoyancy
    ωx_I0 = -bx/y
    ωy_I0 = -by/y
    χx_I0 = @. (z^3 - z)/6 # bx = z
    χy_I0 = @. (z^2 + z)/2 # by = 1
    c1 = -ωx_I0[nz]
    c2 = ωy_I0[nz]
    ωx0_B0 = @. exp(q*z_s)*(c1*cos(q*z_s) + c2*sin(q*z_s))
    ωy0_B0 = @. exp(q*z_s)*(c1*sin(q*z_s) - c2*cos(q*z_s))
    χx0_B2 = @. exp(q*z_s)*(c2*cos(q*z_s) - c1*sin(q*z_s))/(2q^2)
    χy0_B2 = @. exp(q*z_s)*(c1*sin(q*z_s) - c2*cos(q*z_s))/(2q^2)
    c1 = -5q/6 # bx = z
    c2 = q/6 # by = 1
    χx_I1 = @. -c2*z/(2*H*q^2)
    χy_I1 = @. +c1*z/(2*H*q^2)
    ωx_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))
    ωy_B1 = @. exp(-q*z_b)*(c2*cos(q*z_b) - c1*sin(q*z_b))
    χx_B1 = @. exp(-q*z_b)*(c1*sin(q*z_b) - c2*cos(q*z_b))/(2q^2)
    χy_B1 = @. exp(-q*z_b)*(c1*cos(q*z_b) + c2*sin(q*z_b))/(2q^2)
    ωx_BL = ωx_I0 .+ ωx0_B0 .+ 1/ε*ωx_B1
    ωy_BL = ωy_I0 .+ ωy0_B0 .+ 1/ε*ωy_B1
    χx_BL = χx_I0 .+ ε*(χx_I1 .+ χx_B1) .+ ε²*χx0_B2
    χy_BL = χy_I0 .+ ε*(χy_I1 .+ χy_B1) .+ ε²*χy0_B2

    # plot
    fig, ax = plt.subplots(2, 2, figsize=(3.2, 5.2))
    ax[1, 1].plot(ωx, z, label=L"\omega^x")
    ax[1, 1].plot(ωy, z, label=L"\omega^y")
    ax[1, 1].plot(ωx_BL, z, "k--", lw=0.5)
    ax[1, 1].plot(ωy_BL, z, "k--", lw=0.5)
    ax[1, 2].plot(χx, z, label=L"\chi^x")
    ax[1, 2].plot(χy, z, label=L"\chi^y")
    ax[1, 2].plot(χx_BL, z, "k--", lw=0.5)
    ax[1, 2].plot(χy_BL, z, "k--", lw=0.5)
    ax[2, 1].plot(ωx, z, label=L"\omega^x")
    ax[2, 1].plot(ωy, z, label=L"\omega^y")
    ax[2, 1].plot(ωx_BL, z, "k--", lw=0.5)
    ax[2, 1].plot(ωy_BL, z, "k--", lw=0.5)
    ax[2, 2].plot(χx, z, label=L"\chi^x")
    ax[2, 2].plot(χy, z, label=L"\chi^y")
    ax[2, 2].plot(χx_BL, z, "k--", lw=0.5)
    ax[2, 2].plot(χy_BL, z, "k--", lw=0.5)
    ax[1, 1].set_ylabel(L"z")
    ax[2, 1].set_ylabel(L"z")
    ax[2, 1].set_xlabel(L"\omega")
    ax[2, 2].set_xlabel(L"\chi")
    ax[1, 1].legend()
    ax[1, 2].legend()
    ax[2, 1].set_xlim(-2/ε, 2/ε)
    ax[2, 1].set_ylim(-1, -1 + 10*ε/q)
    ax[2, 2].set_xlim(-2*ε, 2*ε)
    ax[2, 2].set_ylim(-1, -1 + 10*ε/q)
    ax[1, 2].set_yticklabels([])
    ax[2, 2].set_yticklabels([])
    savefig("scratch/images/omega_chi.png")
    println("scratch/images/omega_chi.png")
    plt.close()
end