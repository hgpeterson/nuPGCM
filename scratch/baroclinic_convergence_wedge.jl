using nuPGCM
using PyPlot
using Printf
using WriteVTK

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function second_order_wedge_col(p, t)
    # midpts of surface triangle
    p_sfc = p[1:3, 1:2]
    edges = [[1,2], [2,3], [3,1]]
    p_sfc_mids = [sum(p_sfc[edges[i], j])/2 for i=1:3, j=1:2]

    # generate column
    nσ = Int64(size(p, 1)/3)
    σ = p[1:3:end, 3]
    nt = nσ - 1
    t2 = hcat(t, 3nσ .+ [3*(i - 1) + j for i=1:nt, j=1:6])
    p2 = vcat(p, hcat(repeat(p_sfc_mids, nσ), repeat(σ, inner=3)))

    vtk_grid("output/col1.vtu", p2', [MeshCell(VTKCellTypes.VTK_WEDGE, t2[k, 1:6]) for k ∈ axes(t, 1)]) do vtk end
    vtk_grid("output/col2.vtu", p2', [MeshCell(VTKCellTypes.VTK_WEDGE, t2[k, 7:12]) for k ∈ axes(t, 1)]) do vtk end

    return p2, t2
end

function convergence_wedge(h)
    # params 
    ε² = 1e-2
    f = 1

    # surface triangle 
    p_sfc = [0      0
             2h/√3  0
             h/√3   h]

    # node grids
    σ = -1:h:0
    nσ = length(σ)

    # generate column
    nt = nσ - 1
    t = [3*(i - 1) + j for i=1:nt, j=1:6]
    np = sum(3nσ)
    p = hcat(repeat(p_sfc, nσ), repeat(σ, inner=3))
    vtk_grid("output/col.vtu", p', [MeshCell(VTKCellTypes.VTK_WEDGE, t[k, :]) for k ∈ axes(t, 1)]) do vtk end

    # second order buoyancy
    p2, t2 = second_order_wedge_col(p, t)
    # p_sfc2 = vcat(p_sfc, p2[np+1:np+3, 1:2])

    # buoyancy
    # H = [1+h, 1, 1, 1+h/2, 1, 1+h/2]
    H = [1, 1, 1, 1, 1, 1]
    x2 = p2[:, 1]
    y2 = p2[:, 2]
    σ2 = p2[:, 3]
    z2 = σ2#.*vcat(repeat(H[1:3], nσ), repeat(H[4:6], nσ))
    b = @. exp(x2 + y2 + z2)
    bx = [exp(p_sfc[i, 1] + p_sfc[i, 2] + σ[j]*H[i]) for i=1:3, j=1:nσ]
    by = [exp(p_sfc[i, 1] + p_sfc[i, 2] + σ[j]*H[i]) for i=1:3, j=1:nσ] 

    # numerical gradients
    bxₕ = zeros(3, 2nσ-2)
    byₕ = zeros(3, 2nσ-2)
    w1 = Wedge(order=1)
    w2 = Wedge(order=2)
    Dξ = [φξ(w2, w1.p_ref[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dη = [φη(w2, w1.p_ref[i, :], j) for i=1:w1.n, j=1:w2.n]
    Dζ = [φζ(w2, w1.p_ref[i, :], j) for i=1:w1.n, j=1:w2.n]
    for k=1:nt
        for i=1:3
            i1 = i 
            jacobian = J(w1, w1.p_ref[i1, :], p[t[k, :], :])
            ξx = jacobian[1, 1]
            ηx = jacobian[2, 1]
            ζx = jacobian[3, 1]
            ξy = jacobian[1, 2]
            ηy = jacobian[2, 2]
            ζy = jacobian[3, 2]
            bxₕ[i, 2k-1] = sum(b[t2[k, j]]*(Dξ[i1, j]*ξx + Dη[i1, j]*ηx + Dζ[i1, j]*ζx) for j=1:w2.n)
            byₕ[i, 2k-1] = sum(b[t2[k, j]]*(Dξ[i1, j]*ξy + Dη[i1, j]*ηy + Dζ[i1, j]*ζy) for j=1:w2.n)

            i2 = i + 3
            jacobian = J(w1, w1.p_ref[i2, :], p[t[k, :], :])
            ξx = jacobian[1, 1]
            ηx = jacobian[2, 1]
            ζx = jacobian[3, 1]
            ξy = jacobian[1, 2]
            ηy = jacobian[2, 2]
            ζy = jacobian[3, 2]
            bxₕ[i, 2k] = sum(b[t2[k, j]]*(Dξ[i2, j]*ξx + Dη[i2, j]*ηx + Dζ[i2, j]*ζx) for j=1:w2.n)
            byₕ[i, 2k] = sum(b[t2[k, j]]*(Dξ[i2, j]*ξy + Dη[i2, j]*ηy + Dζ[i2, j]*ζy) for j=1:w2.n)
        end
    end

    for i=1:3
        z_dg = zeros(2nσ-2)
        bx_dg = zeros(2nσ-2)
        by_dg = zeros(2nσ-2)
        for j=1:nσ-1
            z_dg[2j-1] = H[i]*σ[j]
            z_dg[2j] = H[i]*σ[j+1]
            bx_dg[2j-1] = bx[i, j]
            bx_dg[2j] = bx[i, j+1]
            by_dg[2j-1] = by[i, j]
            by_dg[2j] = by[i, j+1]
        end
        bx_err = maximum(abs.(bx_dg - bxₕ[i, :]))
        by_err = maximum(abs.(by_dg - byₕ[i, :]))
        println(@sprintf("%1.1e  %1.1e", bx_err, by_err))
        fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
        ax[1].plot(bx[i, :], H[i]*σ)
        ax[1].plot(bxₕ[i, :], z_dg, "--")
        ax[1].set_xlabel(L"\partial_x b")
        ax[1].set_ylabel(L"z")
        ax[2].plot(by[i, :], H[i]*σ)
        ax[2].plot(byₕ[i, :], z_dg, "--")
        ax[2].set_xlabel(L"\partial_y b")
        ax[1].set_title(latexstring(L"i = ", i))
        savefig("scratch/images/bxby$i.png")
        println("scratch/images/bxby$i.png")
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
    for i=1:3
        z = H[i]*σ
        A = nuPGCM.get_baroclinic_LHS(z, ε², f)
        r = nuPGCM.get_baroclinic_RHS(z, bxₕ[i, :], byₕ[i, :], 0, 0, 0, 0, ε²)
        r_a = nuPGCM.get_baroclinic_RHS(z, bx[i, :], by[i, :], 0, 0, 0, 0, ε²)
        sol = A\r
        ωx[(i-1)*nσ+1:i*nσ] = sol[0*nσ+1:1*nσ]
        ωy[(i-1)*nσ+1:i*nσ] = sol[1*nσ+1:2*nσ]
        χx[(i-1)*nσ+1:i*nσ] = sol[2*nσ+1:3*nσ]
        χy[(i-1)*nσ+1:i*nσ] = sol[3*nσ+1:4*nσ]
        sol_a = A\r_a
        ωx_a[(i-1)*nσ+1:i*nσ]= sol_a[0*nσ+1:1*nσ]
        ωy_a[(i-1)*nσ+1:i*nσ]= sol_a[1*nσ+1:2*nσ]
        χx_a[(i-1)*nσ+1:i*nσ]= sol_a[2*nσ+1:3*nσ]
        χy_a[(i-1)*nσ+1:i*nσ]= sol_a[3*nσ+1:4*nσ]

        fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
        ax[1].plot(sol_a[0*nσ+1:1*nσ], z)
        ax[1].plot(sol[0*nσ+1:1*nσ], z, "k--", lw=0.5)
        ax[1].plot(sol_a[1*nσ+1:2*nσ], z)
        ax[1].plot(sol[1*nσ+1:2*nσ], z, "k--", lw=0.5)
        ax[2].plot(sol_a[2*nσ+1:3*nσ], z)
        ax[2].plot(sol[2*nσ+1:3*nσ], z, "k--", lw=0.5)
        ax[2].plot(sol_a[3*nσ+1:4*nσ], z)
        ax[2].plot(sol[3*nσ+1:4*nσ], z, "k--", lw=0.5)
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

convergence_wedge(0.1)

println("Done.")