using nuPGCM
using PyPlot
using Printf
using WriteVTK

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

include("elements.jl")

function second_order_hexa_col(h)
    # surface quad
    p_sfc = [0  0
             h/2  h/2
             0 h
             -h/2 h/2
             h/4 h/4
             h/4 3h/4
             -h/4 3h/4
             -h/4 h/4]
        
    # depths
    H = [1+h, 1+h, 1, 1, 1+h, 1+h/2, 1, 1+h/2]

    # node grids
    nz = Int64(round(1/h))
    nzs = nz*ones(Int64, 8)
    z_cols = [range(0, -H[i], length=nzs[i]) for i ∈ eachindex(nzs)]

    # generate column
    nt = maximum(nzs) - 1
    np = sum(nzs)
    p = zeros(np, 3)
    t = Vector{Vector{Int64}}(undef, nt)
    # start with surface
    top = [1, 2, 3, 4, 5, 6, 7, 8]
    p[top, :] = hcat(p_sfc, zeros(8))
    # iterate down
    i_p = 9
    bot = [0, 0, 0, 0, 0, 0, 0, 0]
    for k=1:nt
        # add bots
        for i=1:8
            p[i_p, :] = vcat(p_sfc[i, 1:2], z_cols[i][k+1])
            bot[i] = i_p
            i_p += 1
        end
        t[k] = vcat(top[1:4], bot[1:4], top[5:8], bot[5:8])
        top[:] = bot[:]
    end

    return p, t
end


function convergence_hexa(h)
    # params 
    ε² = 1e-2
    f = 1

    # surface quad 
    p_sfc = [0  0
             h/2  h/2
             0 h
             -h/2 h/2]

    # depths
    H = [1+h, 1+h, 1, 1]

    # node grids
    nz = Int64(round(1/h))
    nzs = nz*ones(Int64, 4)
    z_cols = [range(0, -H[i], length=nzs[i]) for i ∈ eachindex(nzs)]

    # generate column
    np = sum(nzs)
    nt = maximum(nzs) - 1
    p = zeros(np, 3)
    t = Vector{Vector{Int64}}(undef, nt)
    # start with surface
    top = [1, 2, 3, 4]
    p[top, :] = hcat(p_sfc, zeros(4))
    # iterate down
    i_p = 5
    bot = [0, 0, 0, 0]
    for k=1:nt
        # add bots
        for i=1:4
            p[i_p, :] = vcat(p_sfc[i, 1:2], z_cols[i][k+1])
            bot[i] = i_p
            i_p += 1
        end
        t[k] = vcat(top, bot)
        top[:] = bot[:]
    end

    vtk_grid("output/col.vtu", p', [MeshCell(VTKCellTypes.VTK_HEXAHEDRON, t[k]) for k ∈ eachindex(t)]) do vtk end

    # second order buoyancy
    p2, t2 = second_order_hexa_col(h)

    # buoyancy
    x = p2[:, 1]
    y = p2[:, 2]
    z = p2[:, 3]
    b = @. exp(x + y + z)
    bx = [@. exp(p_sfc[i, 1] + p_sfc[i, 2] + z_cols[i]) for i ∈ eachindex(z_cols)]
    by = [@. exp(p_sfc[i, 1] + p_sfc[i, 2] + z_cols[i]) for i ∈ eachindex(z_cols)] 

    # numerical gradients
    bxₕ = [zeros(2nz-2) for nz ∈ nzs]
    byₕ = [zeros(2nz-2) for nz ∈ nzs]
    el1 = Hexahedron(order=1)
    el2 = Hexahedron(order=2)
    Dξ = [φξ(el2, el1.p_ref[i, :], j) for i ∈ axes(el1.p_ref, 1), j ∈ axes(el2.p_ref, 1)]
    Dη = [φη(el2, el1.p_ref[i, :], j) for i ∈ axes(el1.p_ref, 1), j ∈ axes(el2.p_ref, 1)]
    Dζ = [φζ(el2, el1.p_ref[i, :], j) for i ∈ axes(el1.p_ref, 1), j ∈ axes(el2.p_ref, 1)]
    for k=1:nt
        for i=1:4
            i1 = i 
            jacobian = J(el1, el1.p_ref[i1, :], p[t[k], :])
            ξx = jacobian[1, 1]
            ηx = jacobian[2, 1]
            ζx = jacobian[3, 1]
            ξy = jacobian[1, 2]
            ηy = jacobian[2, 2]
            ζy = jacobian[3, 2]
            bxₕ[i][2k-1] = sum(b[t2[k][j]]*(Dξ[i1, j]*ξx + Dη[i1, j]*ηx + Dζ[i1, j]*ζx) for j ∈ axes(el2.p_ref, 1))
            byₕ[i][2k-1] = sum(b[t2[k][j]]*(Dξ[i1, j]*ξy + Dη[i1, j]*ηy + Dζ[i1, j]*ζy) for j ∈ axes(el2.p_ref, 1))

            i2 = i + 4
            jacobian = J(el1, el1.p_ref[i2, :], p[t[k], :])
            ξx = jacobian[1, 1]
            ηx = jacobian[2, 1]
            ζx = jacobian[3, 1]
            ξy = jacobian[1, 2]
            ηy = jacobian[2, 2]
            ζy = jacobian[3, 2]
            bxₕ[i][2k] = sum(b[t2[k][j]]*(Dξ[i2, j]*ξx + Dη[i2, j]*ηx + Dζ[i2, j]*ζx) for j ∈ axes(el2.p_ref, 1))
            byₕ[i][2k] = sum(b[t2[k][j]]*(Dξ[i2, j]*ξy + Dη[i2, j]*ηy + Dζ[i2, j]*ζy) for j ∈ axes(el2.p_ref, 1))
        end
    end

    for i=1:4
        z_dg = zeros(2nzs[i]-2)
        bx_dg = zeros(2nzs[i]-2)
        by_dg = zeros(2nzs[i]-2)
        for j=1:nzs[i]-1
            z_dg[2j-1] = z_cols[i][j]
            z_dg[2j] = z_cols[i][j+1]
            bx_dg[2j-1] = bx[i][j]
            bx_dg[2j] = bx[i][j+1]
            by_dg[2j-1] = by[i][j]
            by_dg[2j] = by[i][j+1]
        end
        bx_err = maximum(abs.(bx_dg - bxₕ[i]))
        by_err = maximum(abs.(bx_dg - bxₕ[i]))
        println(@sprintf("%1.1e  %1.1e", bx_err, by_err))
        fig, ax = plt.subplots(1, figsize=(2, 3.2))
        ax.plot(bx[i], z_cols[i])
        ax.plot(bxₕ[i], z_dg, "--")
        ax.set_xlabel(L"\partial_x b")
        ax.set_ylabel(L"z")
        savefig("scratch/images/bx$i.png")
        println("scratch/images/bx$i.png")
        plt.close()
        fig, ax = plt.subplots(1, figsize=(2, 3.2))
        ax.plot(by[i], z_cols[i])
        ax.plot(byₕ[i], z_dg, "--")
        ax.set_xlabel(L"\partial_y b")
        ax.set_ylabel(L"z")
        savefig("scratch/images/by$i.png")
        println("scratch/images/by$i.png")
        plt.close()
    end

    # solve
    ωx = zeros(np)
    ωy = zeros(np)
    χx = zeros(np)
    χy = zeros(np)
    ωx_a = zeros(np)
    ωy_a = zeros(np)
    χx_a = zeros(np)
    χy_a = zeros(np)
    for i=1:4
        A = nuPGCM.get_baroclinic_LHS(z_cols[i], ε², f)
        r = nuPGCM.get_baroclinic_RHS(z_cols[i], bxₕ[i], byₕ[i], 0, 0, 0, 0, ε²)
        r_a = nuPGCM.get_baroclinic_RHS(z_cols[i], bx[i], by[i], 0, 0, 0, 0, ε²)
        sol = A\r
        nz = nzs[i]
        ωx[(i-1)*nz+1:i*nz] = sol[0*nz+1:1*nz]
        ωy[(i-1)*nz+1:i*nz] = sol[1*nz+1:2*nz]
        χx[(i-1)*nz+1:i*nz] = sol[2*nz+1:3*nz]
        χy[(i-1)*nz+1:i*nz] = sol[3*nz+1:4*nz]
        sol_a = A\r_a
        ωx_a[(i-1)*nz+1:i*nz]= sol_a[0*nz+1:1*nz]
        ωy_a[(i-1)*nz+1:i*nz]= sol_a[1*nz+1:2*nz]
        χx_a[(i-1)*nz+1:i*nz]= sol_a[2*nz+1:3*nz]
        χy_a[(i-1)*nz+1:i*nz]= sol_a[3*nz+1:4*nz]

        fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
        ax[1].plot(sol_a[0*nz+1:1*nz], z_cols[i])
        ax[1].plot(sol[0*nz+1:1*nz], z_cols[i], "k--", lw=0.5)
        ax[1].plot(sol_a[1*nz+1:2*nz], z_cols[i])
        ax[1].plot(sol[1*nz+1:2*nz], z_cols[i], "k--", lw=0.5)
        ax[2].plot(sol_a[2*nz+1:3*nz], z_cols[i])
        ax[2].plot(sol[2*nz+1:3*nz], z_cols[i], "k--", lw=0.5)
        ax[2].plot(sol_a[3*nz+1:4*nz], z_cols[i])
        ax[2].plot(sol[3*nz+1:4*nz], z_cols[i], "k--", lw=0.5)
        ax[1].set_xlabel(L"\omega")
        ax[2].set_xlabel(L"\chi")
        ax[1].set_ylabel(L"z")
        savefig("scratch/images/omega_chi$i.png")
        println("scratch/images/omega_chi$i.png")
        plt.close()
    end
    ωx_e = maximum(abs.(ωx - ωx_a))/maximum(abs.(ωx_a))
    ωy_e = maximum(abs.(ωy - ωy_a))/maximum(abs.(ωy_a))
    χx_e = maximum(abs.(χx - χx_a))/maximum(abs.(χx_a))
    χy_e = maximum(abs.(χy - χy_a))/maximum(abs.(χy_a))
    println(@sprintf("%1.1e  %1.1e  %1.1e  %1.1e", ωx_e, ωy_e, χx_e, χy_e))
end

convergence_hexa(0.01)

println("Done.")