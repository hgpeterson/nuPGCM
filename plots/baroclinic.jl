using nuPGCM
using PyPlot
using SparseArrays
using Printf

include("derivatives.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)
pc = 1/6

function bl_spiral(z, q, c1, c2)
    pm = z[end] > 0 ? -1 : +1
    return @. exp(pm*q*z)*(c1*cos(q*z) + c2*sin(q*z))
end

function solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    nz = length(z)

    ν_z  = differentiate(ν, z)
    ν_zz = differentiate(ν_z, z)

    A = Tuple{Int64, Int64, Float64}[]
    r = zeros(2nz)
    imap = reshape(1:2nz, 2, :)
    for i ∈ 2:nz-1
        fd_z   = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz  = mkfdstencil(z[i-1:i+1], z[i], 2)

        ## -f τy - ε² (ν τx)_zz = -b_x

        # -f τy
        push!(A, (imap[1, i], imap[2, i], -f))
        # -2 ε² ν_z τx_z
        push!(A, (imap[1, i], imap[1, i-1], -2ε^2*ν_z[i]*fd_z[1]))
        push!(A, (imap[1, i], imap[1, i  ], -2ε^2*ν_z[i]*fd_z[2]))
        push!(A, (imap[1, i], imap[1, i+1], -2ε^2*ν_z[i]*fd_z[3]))
        # -ε² ν τx_zz
        push!(A, (imap[1, i], imap[1, i-1], -ε^2*ν[i]*fd_zz[1]))
        push!(A, (imap[1, i], imap[1, i  ], -ε^2*ν[i]*fd_zz[2]))
        push!(A, (imap[1, i], imap[1, i+1], -ε^2*ν[i]*fd_zz[3]))
        # -ε² ν_zz τx
        push!(A, (imap[1, i], imap[1, i], -ε^2*ν_zz[i]))
        # -b_x
        r[imap[1, i]] = -bx[i]

        ## f τx - ε² (ν τy)_zz = -b_y

        # f τx
        push!(A, (imap[2, i], imap[1, i], f))
        # -2 ε² ν_z τy_z
        push!(A, (imap[2, i], imap[2, i-1], -2ε^2*ν_z[i]*fd_z[1]))
        push!(A, (imap[2, i], imap[2, i  ], -2ε^2*ν_z[i]*fd_z[2]))
        push!(A, (imap[2, i], imap[2, i+1], -2ε^2*ν_z[i]*fd_z[3]))
        # -ε² ν τy_zz
        push!(A, (imap[2, i], imap[2, i-1], -ε^2*ν[i]*fd_zz[1]))
        push!(A, (imap[2, i], imap[2, i  ], -ε^2*ν[i]*fd_zz[2]))
        push!(A, (imap[2, i], imap[2, i+1], -ε^2*ν[i]*fd_zz[3]))
        # -ε² ν_zz τy
        push!(A, (imap[2, i], imap[2, i], -ε^2*ν_zz[i]))
        # -b_x
        r[imap[2, i]] = -by[i]
    end

    # boundary conditions
    push!(A, (imap[1, nz], imap[1, nz], ε^2*ν[nz]))
    r[imap[1, nz]] = τx
    push!(A, (imap[2, nz], imap[2, nz], ε^2*ν[nz]))
    r[imap[2, nz]] = τy
    for i ∈ 1:nz-1
        push!(A, (imap[1, 1], imap[1, i],   z[i]*(z[i+1] - z[i])/2))
        push!(A, (imap[1, 1], imap[1, i+1], z[i]*(z[i+1] - z[i])/2))
        push!(A, (imap[2, 1], imap[2, i],   z[i]*(z[i+1] - z[i])/2))
        push!(A, (imap[2, 1], imap[2, i+1], z[i]*(z[i+1] - z[i])/2))
    end
    r[imap[1, 1]] = -U
    r[imap[2, 1]] = -V

    # solve
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), 2nz, 2nz)
    sol = A\r
    τx = sol[imap[1, :]]
    τy = sol[imap[2, :]]

    return cumtrapz(τx, z), cumtrapz(τy, z)
end

function solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)
    H = -z[1]

    # interior O(1)
    uI0_bot = U/H - 1/(H*f)*trapz(z.*by, z) - τy/(H*f)
    vI0_bot = V/H + 1/(H*f)*trapz(z.*bx, z) + τx/(H*f)
    uI0 = uI0_bot .- 1/f*cumtrapz(by, z)
    vI0 = vI0_bot .+ 1/f*cumtrapz(bx, z)

    # bottom BL O(1)
    q_b = sqrt(f/2/ν[1])
    z_b = (z .+ H)/ε
    c1 = -uI0_bot
    c2 = -vI0_bot
    uB0_b = bl_spiral(z_b, q_b, c1,  c2)
    vB0_b = bl_spiral(z_b, q_b, c2, -c1)

    # surface BL O(1/ε)
    q_s = sqrt(f/2/ν[end])
    z_s = z/ε
    c1 = (τx + τy)/(2ν[end]*q_s)
    c2 = (τx - τy)/(2ν[end]*q_s)
    uB0_s = bl_spiral(z_s, q_s,  c1, c2)
    vB0_s = bl_spiral(z_s, q_s, -c2, c1)

    # interior O(ε)
    uI1 =  (uI0_bot + vI0_bot)/(2H*q_b)
    vI1 = -(uI0_bot - vI0_bot)/(2H*q_b)

    # bottom BL O(ε)
    c1 = -uI1
    c2 = -vI1
    uB1_b = bl_spiral(z_b, q_b, c1,  c2)
    vB1_b = bl_spiral(z_b, q_b, c2, -c1)

    return 1/ε*uB0_s .+ uI0 .+ uB0_b .+ ε*(uI1 .+ uB1_b),
           1/ε*vB0_s .+ vI0 .+ vB0_b .+ ε*(vI1 .+ vB1_b)
end

function baroclinic_with_wind_stress()
    # params
    ε = 1e-2
    f = 1
    H = 1
    nz = 2^8
    z = -H*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
    ν = ones(nz)
    # ν = @. 1e-2 + exp(-(z + H)/0.1)

    # transport
    τx = 0
    τy = 0
    U = H^2
    V = 0
    bx = zeros(nz)
    by = zeros(nz)
    uU, vU = solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    uUBL, vUBL = solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)

    # wind stress
    τx = 1
    τy = 0
    U = 0
    V = 0
    bx = zeros(nz)
    by = zeros(nz)
    uτ, vτ = solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    uτBL, vτBL = solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)

    # buoyancy
    τx = 0
    τy = 0
    U = 0
    V = 0
    bx = @. exp(-(z + H)/0.1)
    by = zeros(nz)
    ub, vb = solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    ubBL, vbBL = solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)

    # bottom stress stats
    qb = sqrt(f/2/ν[1])
    @printf("∂z(uU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uU[1:3], z[1:3], z[1], 1),  qb/ε*H)
    @printf("∂z(vU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vU[1:3], z[1:3], z[1], 1),  qb/ε*H)
    @printf("∂z(uτ)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uτ[1:3], z[1:3], z[1], 1), -qb/ε*1/(H*f))
    @printf("∂z(vτ)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vτ[1:3], z[1:3], z[1], 1),  qb/ε*1/(H*f))
    @printf("∂z(ub)(-H) = % .2e (% .2e)\n", differentiate_pointwise(ub[1:3], z[1:3], z[1], 1), -qb/ε*trapz(z.*bx, z)/(H*f))
    @printf("∂z(vb)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vb[1:3], z[1:3], z[1], 1),  qb/ε*trapz(z.*bx, z)/(H*f))

    fig, ax = plt.subplots(1, 3, figsize=(27pc, 16pc))
    for a ∈ ax
        a.spines["left"].set_visible(false)
        a.spines["top"].set_visible(true)
        a.axvline(0, lw=0.5, c="k")
        a.set_xlabel(L"Velocity $u$, $v$")
    end
    ax[1].text(-0.04, 1.05, s="(a)", transform=ax[1].transAxes, ha="center")
    ax[2].text(-0.04, 1.05, s="(b)", transform=ax[2].transAxes, ha="center")
    ax[3].text(-0.04, 1.05, s="(c)", transform=ax[3].transAxes, ha="center")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[1].plot(uU, z)
    ax[1].plot(vU, z)
    ax[1].plot(uUBL, z, "k--", lw=0.5)
    ax[1].plot(vUBL, z, "k--", lw=0.5)
    ax[1].set_title(L"$U = H^2$")
    ax[2].plot(uτ, z, label=L"u")
    ax[2].plot(vτ, z, label=L"v")
    ax[2].plot(uτBL, z, "k--", lw=0.5, label="Theory")
    ax[2].plot(vτBL, z, "k--", lw=0.5)
    ax[2].set_title(L"$\tau^x = 1$")
    ax[2].legend(loc=(0.55, 0.6))
    # ax[2].set_xlim(-10, 10)
    # ax[2].set_ylim(-0.1, 0.0)
    ax[3].plot(ub, z)
    ax[3].plot(vb, z)
    ax[3].plot(ubBL, z, "k--", lw=0.5)
    ax[3].plot(vbBL, z, "k--", lw=0.5)
    ax[3].set_title(L"$\partial_x b = e^{-(z + H)/0.1}$")
    savefig("baroclinic.png")
    println("baroclinic.png")
    savefig("baroclinic.pdf")
    println("baroclinic.pdf")
    plt.close()
end

function baroclinic()
    # params
    ε = 1e-2
    f = 1
    H = 1
    nz = 2^8
    z = -H*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
    ν = ones(nz)
    # ν = @. 1e-2 + exp(-(z + H)/0.1)

    # U transport
    τx = 0
    τy = 0
    U = 1
    V = 0
    bx = zeros(nz)
    by = zeros(nz)
    uU, vU = solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    uUBL, vUBL = solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)

    # V transport
    τx = 0
    τy = 0
    U = 0
    V = 1
    bx = zeros(nz)
    by = zeros(nz)
    uV, vV = solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    uVBL, vVBL = solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)

    # buoyancy
    τx = 0
    τy = 0
    U = 0
    V = 0
    bx = @. exp(-(z + H)/0.1)
    by = zeros(nz)
    ub, vb = solve_baroclinic_problem(ε, z, ν, f, bx, by, U, V, τx, τy)
    ubBL, vbBL = solve_baroclinic_problem_BL(ε, z, ν, f, bx, by, U, V, τx, τy)

    # bottom stress stats
    qb = sqrt(f/2/ν[1])
    @printf("∂z(uU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uU[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(vU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vU[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(uV)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uV[1:3], z[1:3], z[1], 1), -qb/ε/H)
    @printf("∂z(vV)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vV[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(ub)(-H) = % .2e (% .2e)\n", differentiate_pointwise(ub[1:3], z[1:3], z[1], 1),  -qb/ε*trapz(z.*bx, z)/(H*f))
    @printf("∂z(vb)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vb[1:3], z[1:3], z[1], 1),   qb/ε*trapz(z.*bx, z)/(H*f))

    width = 27pc
    fig, ax = plt.subplots(1, 3, figsize=(width, width/3*1.62))
    for a ∈ ax
        a.spines["left"].set_visible(false)
        a.axvline(0, lw=0.5, c="k")
        a.set_xlabel(L"Velocity $u$, $v$")
    end
    ax[1].text(-0.04, 1.05, s="(a)", transform=ax[1].transAxes, ha="center")
    ax[2].text(-0.04, 1.05, s="(b)", transform=ax[2].transAxes, ha="center")
    ax[3].text(-0.04, 1.05, s="(c)", transform=ax[3].transAxes, ha="center")
    ax[1].set_xlim(-0.5, 1.5)
    ax[2].set_xlim(-0.5, 1.5)
    ax[3].set_xlim(-0.08, 0.08)
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[1].plot(uU, z, label=L"u")
    ax[1].plot(vU, z, label=L"v")
    ax[1].plot(uUBL, z, "k--", lw=0.5, label="Theory")
    ax[1].plot(vUBL, z, "k--", lw=0.5)
    ax[1].set_title(L"$U = 1, \; V = 0$")
    ax[2].plot(uV, z, label=L"u")
    ax[2].plot(vV, z, label=L"v")
    ax[2].plot(uVBL, z, "k--", lw=0.5, label="Theory")
    ax[2].plot(vVBL, z, "k--", lw=0.5)
    ax[2].set_title(L"$U = 0, \; V = 1$")
    ax[3].plot(ub, z, label=L"u")
    ax[3].plot(vb, z, label=L"v")
    ax[3].plot(ubBL, z, "k--", lw=0.5, label="Theory")
    ax[3].plot(vbBL, z, "k--", lw=0.5)
    ax[3].set_title(L"$\partial_x b = e^{-(z + H)/0.1}$")
    ax[3].legend(loc=(-0.2, 0.6))
    savefig("baroclinic.png")
    println("baroclinic.png")
    savefig("baroclinic.pdf")
    println("baroclinic.pdf")
    plt.close()
end

baroclinic()

println("Done.")