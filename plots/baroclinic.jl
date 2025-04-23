using nuPGCM
using JLD2
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

function solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V=nothing, τx, τy, vz_bot=nothing)
    if V === nothing && vz_bot === nothing
        throw(ArgumentError("Either V or vz_bot must be provided."))
    end

    nz = length(z)

    ν_z  = differentiate(ν, z)
    ν_zz = differentiate(ν_z, z)

    A = Tuple{Int64, Int64, Float64}[]
    r = zeros(2nz)
    imap = reshape(1:2nz, 2, :)
    for i ∈ 2:nz-1
        fd_z  = mkfdstencil(z[i-1:i+1], z[i], 1)
        fd_zz = mkfdstencil(z[i-1:i+1], z[i], 2)

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
        if V !== nothing
            push!(A, (imap[2, 1], imap[2, i],   z[i]*(z[i+1] - z[i])/2))
            push!(A, (imap[2, 1], imap[2, i+1], z[i]*(z[i+1] - z[i])/2))
        end
    end
    r[imap[1, 1]] = -U
    if V !== nothing
        r[imap[2, 1]] = -V
    else
        push!(A, (imap[2, 1], imap[2, 1], 1))
        r[imap[2, 1]] = vz_bot
    end


    # solve
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), 2nz, 2nz)
    sol = A\r
    τx = sol[imap[1, :]]
    τy = sol[imap[2, :]]

    return cumtrapz(τx, z), cumtrapz(τy, z)
end

function solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=1/2)
    H = -z[1]

    # surface BL coords
    z_s = z/ε
    q_s = sqrt(f/2/ν[end])

    # bottom BL coords
    z_b = (z .+ H)/ε
    q_b = sqrt(f/2/ν[1])
    println("α = 0.0: q_bot = ", q_b)
    q_b *= (1 + α^2*Hx^2)^(-3/4)
    println("α = 0.5: q_bot = ", q_b)

    # interior O(1)
    wI0_sfc = 0 # for now
    uI0_bot = U/H - 1/(H*f)*trapz(z.*by, z) - τy/(H*f)
    vI0_bot = V/H + 1/(H*f)*trapz(z.*bx, z) + τx/(H*f)
    uI0 = uI0_bot .- 1/f*cumtrapz(by, z)
    vI0 = vI0_bot .+ 1/f*cumtrapz(bx, z)
    wI0 = wI0_sfc .- β/f*(trapz(vI0, z) .- cumtrapz(vI0, z))

    # bottom BL O(1)
    c1 = -uI0_bot
    c2 = -vI0_bot
    uB0_b = bl_spiral(z_b, q_b, c1,  c2)
    vB0_b = bl_spiral(z_b, q_b, c2, -c1)
    wB0_b = -Hx*uB0_b - Hy*vB0_b

    # surface BL O(1/ε)
    c1 = (τx + τy)/(2ν[end]*q_s)
    c2 = (τx - τy)/(2ν[end]*q_s)
    uBm1_s = bl_spiral(z_s, q_s,  c1, c2)
    vBm1_s = bl_spiral(z_s, q_s, -c2, c1)

    # interior O(ε)
    uI1 =  (uI0_bot + vI0_bot)/(2H*q_b)
    vI1 = -(uI0_bot - vI0_bot)/(2H*q_b)
    wI1 = -β/f*(H*vI1 .- cumtrapz(vI1*ones(length(z)), z))

    # bottom BL O(ε)
    c1 = -uI1
    c2 = -vI1
    uB1_b = bl_spiral(z_b, q_b, c1,  c2)
    vB1_b = bl_spiral(z_b, q_b, c2, -c1)
    wB1_b = -Hx*uB1_b - Hy*vB1_b

    # surface BL O(ε)
    c1 = -(bx[end] - by[end])/(2*f*q_s)
    c2 = +(bx[end] + by[end])/(2*f*q_s)
    uB1_s = bl_spiral(z_s, q_s,  c1, c2)
    vB1_s = bl_spiral(z_s, q_s, -c2, c1)

    # interior O(ε²)
    uI2_bot = ν[1]/H/f^2*bx[1] + 1/f^2*differentiate_pointwise(ν[1:3].*bx[1:3], z[1:3], z[1], 1) + (uI1 + vI1)/(2q_b*H)
    vI2_bot = ν[1]/H/f^2*by[1] + 1/f^2*differentiate_pointwise(ν[1:3].*by[1:3], z[1:3], z[1], 1) - (uI1 - vI1)/(2q_b*H)
    uI2 = uI2_bot .+ 1/f^2*differentiate(ν.*bx, z) .- 1/f^2*differentiate_pointwise(ν[1:3].*bx[1:3], z[1:3], z[1], 1)
    vI2 = vI2_bot .+ 1/f^2*differentiate(ν.*by, z) .- 1/f^2*differentiate_pointwise(ν[1:3].*by[1:3], z[1:3], z[1], 1)
    # wI2 = -β/f*(trapz(vI2, z) .- cumtrapz(vI2, z)) # this is missing the ∂z(ν ∂z(ζ)) term

    # bottom BL O(ε²)
    c1 = -uI2_bot
    c2 = -vI2_bot
    uB2_b = bl_spiral(z_b, q_b, c1,  c2)
    vB2_b = bl_spiral(z_b, q_b, c2, -c1)
    # wB2_b = -Hx*uB2_b - Hy*vB2_b

    # total
    uBL = 1/ε*uBm1_s .+ uI0 .+ uB0_b .+ ε*(uI1 .+ uB1_b .+ uB1_s) .+ ε^2*(uI2 .+ uB2_b)
    vBL = 1/ε*vBm1_s .+ vI0 .+ vB0_b .+ ε*(vI1 .+ vB1_b .+ vB1_s) .+ ε^2*(vI2 .+ vB2_b)
    wBL =               wI0 .+ wB0_b .+ ε*(wI1 .+ wB1_b)          #.+ ε^2*(wI2 .+ wB2_b)

    return uBL, vBL, wBL
end

function solve_baroclinic_problem_BL_U0(; ε, z, ν, f, bx, Hx, α=1/2)
    H = -z[1]
    q_b = sqrt(f/2/ν[1])
    println("α = 0.0: q_bot = ", q_b)
    Γ = 1 + α^2*Hx^2
    q_b *= Γ^(-3/4)
    println("α = 0.5: q_bot = ", q_b)
    z_b = (z .+ H)/ε

    # interior O(1)
    # uI0 = 0
    vI0 = 1/f*cumtrapz(bx, z)
    # wI0 = 0

    # interior O(ε)
    # uI1 = 0
    vI1 = -1/(f*q_b)*bx[1]
    # wI1 = 0

    # bottom BL O(ε)
    c1 = 0
    c2 = -vI1/√Γ
    uB1 =    bl_spiral(z_b, q_b, c1,  c2)
    vB1 = √Γ*bl_spiral(z_b, q_b, c2, -c1)
    wB1 = -Hx*uB1

    # interior O(ε²)
    uI2 =  Γ/f^2*differentiate(ν.*bx, z)
    vI2 = -uI2[1]*√Γ
    wI2 = -Hx*uI2

    # bottom BL O(ε²)
    c1 = -uI2[1]
    c2 = -vI2/√Γ
    uB2 =    bl_spiral(z_b, q_b, c1,  c2)
    vB2 = √Γ*bl_spiral(z_b, q_b, c2, -c1)
    wB2 = -Hx*uB2

    # total
    uBL = ε*uB1 + ε^2*(uI2 .+ uB2)
    vBL = vI0 .+ ε*(vI1 .+ vB1) .+ ε^2*(vI2 .+ vB2)
    wBL = ε*wB1 + ε^2*(wI2 .+ wB2)

    return uBL, vBL, wBL
end

function baroclinic()
    # params
    ε = 2e-2
    f = 1
    β = 0
    H = 0.75
    Hx = -1
    Hy = 0
    nz = 2^10
    z = -H*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
    ν = ones(nz)

    # U transport
    τx = 0
    τy = 0
    U = 1
    V = 0
    bx = zeros(nz)
    by = zeros(nz)
    uU, vU = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V, τx, τy)
    uUBL, vUBL, wUBL = solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)

    # V transport
    τx = 0
    τy = 0
    U = 0
    V = 1
    bx = zeros(nz)
    by = zeros(nz)
    uV, vV = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V, τx, τy)
    uVBL, vVBL, wVBL = solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)

    # buoyancy
    τx = 0
    τy = 0
    U = 0
    V = 0
    d = jldopen("buoyancy_eps2e-02_T1e-02.jld2", "r")
    x = d["x"]
    i = argmin(abs.(x .- 0.5)) # index where x = 0.5
    z_sim = d["z"][i, :]
    bx = d["bx"][i, :]
    close(d)
    by = zeros(length(z_sim))
    ν_sim = ones(length(z_sim))
    ub, vb = solve_baroclinic_problem(; ε, z=z_sim, ν=ν_sim, f, bx, by, U, V, τx, τy)
    ubBL, vbBL, wbBL = solve_baroclinic_problem_BL(; ε, z=z_sim, ν=ν_sim, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)

    # bottom stress stats
    qb = sqrt(f/2/ν_sim[1])
    @printf("∂z(uU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uU[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(vU)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vU[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(uV)(-H) = % .2e (% .2e)\n", differentiate_pointwise(uV[1:3], z[1:3], z[1], 1), -qb/ε/H)
    @printf("∂z(vV)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vV[1:3], z[1:3], z[1], 1),  qb/ε/H)
    @printf("∂z(ub)(-H) = % .2e (% .2e)\n", differentiate_pointwise(ub[1:3], z_sim[1:3], z_sim[1], 1),  -qb/ε*trapz(z_sim.*bx, z_sim)/(H*f))
    @printf("∂z(vb)(-H) = % .2e (% .2e)\n", differentiate_pointwise(vb[1:3], z_sim[1:3], z_sim[1], 1),   qb/ε*trapz(z_sim.*bx, z_sim)/(H*f))

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
    ax[1].set_xticks([0, 1/H])
    ax[1].set_xticklabels(["0", L"$1/H$"])
    ax[2].set_xlim(-0.5, 1.5)
    ax[2].set_xticks([0, 1/H])
    ax[2].set_xticklabels(["0", L"$1/H$"])
    ax[3].set_xlim(-0.08, 0.08)
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[1].plot(uU, z, label=L"u")
    ax[1].plot(vU, z, label=L"v")
    ax[1].plot(uUBL, z, "k--", lw=0.5, label="BL theory")
    ax[1].plot(vUBL, z, "k--", lw=0.5)
    ax[1].set_title(L"$U = 1, \; V = 0$")
    ax[2].plot(uV, z, label=L"u")
    ax[2].plot(vV, z, label=L"v")
    ax[2].plot(uVBL, z, "k--", lw=0.5, label="BL theory")
    ax[2].plot(vVBL, z, "k--", lw=0.5)
    ax[2].set_title(L"$U = 0, \; V = 1$")
    ax[3].plot(ub, z_sim, label=L"u")
    ax[3].plot(vb, z_sim, label=L"v")
    ax[3].plot(ubBL, z_sim, "k--", lw=0.5, label="BL theory")
    ax[3].plot(vbBL, z_sim, "k--", lw=0.5)
    ax[3].set_title(L"$\partial_x b \neq 0$")
    ax[3].legend(loc=(0.55, 0.7))
    savefig("baroclinic.png")
    println("baroclinic.png")
    savefig("baroclinic.pdf")
    println("baroclinic.pdf")
    plt.close()
end

function wBL()
    # params
    ε = 1e-2
    f = 1
    β = 1
    t = 3e-3
    Hx = -1
    Hy = 0
    τx = 0
    τy = 0
    U = 0
    V = 0

    # load data
    d = jldopen(@sprintf("../sims/sim048/data/1D_b_%1.1e.jld2", t), "r")
    b = d["b"]
    z = d["z"]
    close(d)
    nz = length(z)
    ν = ones(nz)
    bx = Hx*differentiate(b, z)
    by = Hy*differentiate(b, z)

    # solve
    u, v = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, V, τx, τy)
    uBL, vBL, wBL = solve_baroclinic_problem_BL(; ε, z, ν, f, β, bx, by, U, V, τx, τy, Hx, Hy, α=0)
    @printf("max |u - uBL| = %1.1e\n", maximum(abs.(u - uBL)))
    @printf("max |v - vBL| = %1.1e\n", maximum(abs.(v - vBL)))

    width = 27pc
    fig, ax = plt.subplots(1, 3, figsize=(width, width/3*1.62))
    for a ∈ ax
        a.spines["left"].set_visible(false)
        a.axvline(0, lw=0.5, c="k")
    end
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"Zonal flow $u$")
    ax[2].set_xlabel(L"Meridional flow $v$")
    ax[3].set_xlabel(L"Vertical flow $w$")
    ax[1].text(-0.04, 1.05, s="(a)", transform=ax[1].transAxes, ha="center")
    ax[2].text(-0.04, 1.05, s="(b)", transform=ax[2].transAxes, ha="center")
    ax[3].text(-0.04, 1.05, s="(c)", transform=ax[3].transAxes, ha="center")
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    ax[3].set_yticks([])
    ax[1].plot(u, z)
    ax[1].plot(uBL, z, "k--", lw=0.5, label="BL theory")
    ax[1].legend()
    ax[2].plot(v, z)
    ax[2].plot(vBL, z, "k--", lw=0.5, label="BL theory")
    ax[3].plot(wBL, z, "k--", lw=0.5, label="BL theory")
    savefig("wBL.png")
    println("wBL.png")
    plt.close()
end

function test_U0()
    # params
    ε = 2e-2
    f = 1
    β = 0
    Hx = -1
    Hy = 0
    τx = 0
    τy = 0
    U = 0
    vz_bot = 0

    # load bx
    d = jldopen("buoyancy_eps2e-02_T1e-02.jld2", "r")
    x = d["x"]
    z = d["z"]
    bx = d["bx"]
    close(d)
    nz = size(z, 2)
    i = argmin(abs.(x .- 0.5)) # index where x = 0.5
    println("Closest point to x=0.5: x=", x[i])
    z = z[i, :] 
    bx = bx[i, :]
    println(bx[end])
    by = zeros(nz)
    ν = ones(nz)
    u, v = solve_baroclinic_problem(; ε, z, ν, f, bx, by, U, τx, τy, vz_bot)
    uBL, vBL, wBL = solve_baroclinic_problem_BL_U0(; ε, z, ν, f, bx, Hx, α=0)
    fig, ax = plt.subplots(1, 2, figsize=(4, 3.2))
    ax[1].spines["left"].set_visible(false)
    ax[2].spines["left"].set_visible(false)
    ax[1].axvline(0, lw=0.5, c="k")
    ax[2].axvline(0, lw=0.5, c="k")
    ax[1].set_ylabel(L"Vertical coordinate $z$")
    ax[1].set_xlabel(L"$u$")
    ax[2].set_xlabel(L"$v$")
    ax[1].set_yticks([-0.75, -0.5, -0.25, 0])
    ax[2].set_yticks([])
    # ax[1].set_ylim(-0.75, -.7)
    # ax[2].set_ylim(-0.75, -.7)
    ax[1].plot(u, z)
    ax[1].plot(uBL, z, "k--", lw=0.5)
    ax[2].plot(v, z)
    ax[2].plot(vBL, z, "k--", lw=0.5)
    savefig("test_U0.png")
    println("test_U0.png")
    plt.close()
end

# baroclinic()
# wBL()
# test_U0()
# println("Done.")