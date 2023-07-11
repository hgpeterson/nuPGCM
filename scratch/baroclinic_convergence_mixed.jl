using nuPGCM
using PyPlot
using Printf
using WriteVTK

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

include("elements.jl")

function second_order_mixed_col(p_sfc, z_cols, nzs, p, t)
    p_sfc_mid = (p_sfc[1:3, :] + p_sfc[[2,3,1], :])/2
    z_cols_mid = [zeros(max(length(z_cols[i]), length(z_cols[mod1(i+1, 3)]))) for i=1:3]
    for i=1:3
        nzs = [length(z_cols[i]), length(z_cols[mod1(i+1, 3)])]
        i_min = argmin(nzs)
        n_min = nzs[i_min]
        n_max = nzs[mod1(i_min+1, 2)]
        for j=1:n_min
            z_cols_mid[i][j] = (z_cols[i][j] + z_cols[mod1(i+1, 3)][j])/2
        end
        if n_min < n_max
            for j=n_min+1:n_max
                if i_min == 1
                    z_cols_mid[i][j] = (z_cols[i][end] + z_cols[mod1(i+1, 3)][j])/2
                else
                    z_cols_mid[i][j] = (z_cols[i][j] + z_cols[mod1(i+1, 3)][end])/2
                end
            end
        end
    end
    nzs_mid = [length(z) for z ∈ z_cols_mid]

    # generate column
    nt = length(t)
    np = size(p, 1)
    np2 = np + sum(nzs_mid)
    p2 = zeros(np2, 3)
    p2[1:np, :] = p
    t2 = Vector{Vector{Int64}}(undef, nt)
    # start with surface
    top = np .+ [1, 2, 3]
    p2[top, :] = hcat(p_sfc_mid, zeros(3))
    # iterate down
    i_p = np + 4
    bot = copy(top)
    for k=1:nt
        has_more_nodes = findall(nzs_mid .≥ k+1)
        for i ∈ has_more_nodes
            p2[i_p, :] = vcat(p_sfc_mid[i, 1:2], z_cols_mid[i][k+1])
            bot[i] = i_p
            i_p += 1
        end
        if length(t[k]) == 6
            # wedge
            t2[k] = vcat(t[k], top, bot[has_more_nodes])
        elseif length(t[k]) == 5
            # pyramid
            has_no_more_nodes = findfirst(nzs .< k+1)
            println(has_more_nodes)
            has_more_nodes = mod1.((1:3) .+ (has_no_more_nodes - 1), 3) # ensure correct ordering
            println(has_more_nodes)
            t2[k] = vcat(t[k], top[has_more_nodes], bot[has_more_nodes])
        elseif length(t[k]) == 4
            # tetra
            has_no_more_nodes = findfirst(nzs_mid .< k+1)
            if has_no_more_nodes == 1
                has_more_nodes = [3, 2] # ensure correct ordering
            end
            t2[k] = vcat(t[k], top[has_more_nodes[1]], top[has_no_more_nodes], top[has_more_nodes[2]], bot[has_more_nodes])
        end
        top[:] = bot[:]
    end

    # vtk_grid("output/col1.vtu", p', [MeshCell(VTKCellTypes.VTK_WEDGE, t[k][1:6]) for k ∈ eachindex(t)]) do vtk end
    # vtk_grid("output/col2.vtu", p', [MeshCell(VTKCellTypes.VTK_WEDGE, t[k][7:12]) for k ∈ eachindex(t)]) do vtk end

    return p2, t2
end

function convergence_mixed(h)
    # params 
    ε² = 1e-2
    f = 1

    # surface triangle 
    p_sfc = [0      0
             2h/√3  0
             h/√3   h]

    # depths
    H = [h, 0, h]

    # node grids
    nz = Int64(round(1/h))
    nzs = [2, 1, 2]
    # nzs = [nz+2, nz, nz]
    z_cols = [range(0, -H[i], length=nzs[i]) for i ∈ eachindex(H)]
    # z_cols = [0:-h:-H[i] for i ∈ eachindex(H)]
    # nzs = [length(z) for z ∈ z_cols]

    # generate column
    np = sum(nzs)
    nt = maximum(nzs) - 1
    p = zeros(np, 3)
    t = Vector{Vector{Int64}}(undef, nt)
    # start with surface
    top = [1, 2, 3]
    p[top, :] = hcat(p_sfc, zeros(3))
    # iterate down
    i_p = 4
    bot = copy(top)
    for k=1:nt
        # add bots
        has_more_nodes = findall(nzs .≥ k+1)
        for i ∈ has_more_nodes
            p[i_p, :] = vcat(p_sfc[i, 1:2], z_cols[i][k+1])
            bot[i] = i_p
            i_p += 1
        end
        if length(has_more_nodes) == 3
            t[k] = vcat(top, bot)
        elseif length(has_more_nodes) == 2
            has_no_more_nodes = findfirst(nzs .< k+1)
            t[k] = [top[has_more_nodes[1]], bot[has_more_nodes[1]], bot[has_more_nodes[2]], top[has_more_nodes[2]], top[has_no_more_nodes]]
        elseif length(has_more_nodes) == 1
            has_no_more_nodes = findall(nzs .< k+1)
            t[k] = vcat(top[has_more_nodes], top[has_no_more_nodes], bot[has_more_nodes])
        end
        top[:] = bot[:]
    end

    vtk_grid("output/col.vtu", p', [MeshCell(length(t[k]) == 6 ? VTKCellTypes.VTK_WEDGE : (length(t[k]) == 5 ? VTKCellTypes.VTK_PYRAMID : VTKCellTypes.VTK_TETRA), t[k]) for k ∈ eachindex(t)]) do vtk end

    # second order buoyancy
    p2, t2 = second_order_mixed_col(p_sfc, z_cols, nzs, p, t)

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
    w1 = Wedge(order=1)
    w2 = Wedge(order=2)
    Dξ_w = [φξ(w2, w1.p_ref[i, :], j) for i ∈ axes(w1.p_ref, 1), j ∈ axes(w2.p_ref, 1)]
    Dη_w = [φη(w2, w1.p_ref[i, :], j) for i ∈ axes(w1.p_ref, 1), j ∈ axes(w2.p_ref, 1)]
    Dζ_w = [φζ(w2, w1.p_ref[i, :], j) for i ∈ axes(w1.p_ref, 1), j ∈ axes(w2.p_ref, 1)]
    tet1 = Tetra(order=1)
    tet2 = Tetra(order=2)
    Dξ_tet = [φξ(tet2, tet1.p_ref[i, :], j) for i ∈ axes(tet1.p_ref, 1), j ∈ axes(tet2.p_ref, 1)]
    Dη_tet = [φη(tet2, tet1.p_ref[i, :], j) for i ∈ axes(tet1.p_ref, 1), j ∈ axes(tet2.p_ref, 1)]
    Dζ_tet = [φζ(tet2, tet1.p_ref[i, :], j) for i ∈ axes(tet1.p_ref, 1), j ∈ axes(tet2.p_ref, 1)]
    for k=1:nt
        if length(t[k]) == 6
            for i=1:3
                i1 = i 
                jacobian = J(w1, w1.p_ref[i1, :], p[t[k], :])
                ξx = jacobian[1, 1]
                ηx = jacobian[2, 1]
                ζx = jacobian[3, 1]
                ξy = jacobian[1, 2]
                ηy = jacobian[2, 2]
                ζy = jacobian[3, 2]
                bxₕ[i][2k-1] = sum(b[t2[k][j]]*(Dξ_w[i1, j]*ξx + Dη_w[i1, j]*ηx + Dζ_w[i1, j]*ζx) for j ∈ axes(w2.p_ref, 1))
                byₕ[i][2k-1] = sum(b[t2[k][j]]*(Dξ_w[i1, j]*ξy + Dη_w[i1, j]*ηy + Dζ_w[i1, j]*ζy) for j ∈ axes(w2.p_ref, 1))

                i2 = i + 3
                jacobian = J(w1, w1.p_ref[i2, :], p[t[k], :])
                ξx = jacobian[1, 1]
                ηx = jacobian[2, 1]
                ζx = jacobian[3, 1]
                ξy = jacobian[1, 2]
                ηy = jacobian[2, 2]
                ζy = jacobian[3, 2]
                bxₕ[i][2k] = sum(b[t2[k][j]]*(Dξ_w[i2, j]*ξx + Dη_w[i2, j]*ηx + Dζ_w[i2, j]*ζx) for j ∈ axes(w2.p_ref, 1))
                byₕ[i][2k] = sum(b[t2[k][j]]*(Dξ_w[i2, j]*ξy + Dη_w[i2, j]*ηy + Dζ_w[i2, j]*ζy) for j ∈ axes(w2.p_ref, 1))
            end
        elseif length(t[k]) == 4
            i = argmax(nzs)
            i1 = 1
            jacobian = J(tet1, tet1.p_ref[i1, :], p[t[k], :])
            ξx = jacobian[1, 1]
            ηx = jacobian[2, 1]
            ζx = jacobian[3, 1]
            ξy = jacobian[1, 2]
            ηy = jacobian[2, 2]
            ζy = jacobian[3, 2]
            bxₕ[i][2k-1] = sum(b[t2[k][j]]*(Dξ_tet[i1, j]*ξx + Dη_tet[i1, j]*ηx + Dζ_tet[i1, j]*ζx) for j ∈ axes(tet2.p_ref, 1))
            byₕ[i][2k-1] = sum(b[t2[k][j]]*(Dξ_tet[i1, j]*ξy + Dη_tet[i1, j]*ηy + Dζ_tet[i1, j]*ζy) for j ∈ axes(tet2.p_ref, 1))
            i2 = 4
            jacobian = J(tet1, tet1.p_ref[i2, :], p[t[k], :])
            ξx = jacobian[1, 1]
            ηx = jacobian[2, 1]
            ζx = jacobian[3, 1]
            ξy = jacobian[1, 2]
            ηy = jacobian[2, 2]
            ζy = jacobian[3, 2]
            bxₕ[i][2k] = sum(b[t2[k][j]]*(Dξ_tet[i2, j]*ξx + Dη_tet[i2, j]*ηx + Dζ_tet[i2, j]*ζx) for j ∈ axes(tet2.p_ref, 1))
            byₕ[i][2k] = sum(b[t2[k][j]]*(Dξ_tet[i2, j]*ξy + Dη_tet[i2, j]*ηy + Dζ_tet[i2, j]*ζy) for j ∈ axes(tet2.p_ref, 1))
        end
    end

    # plot gradients
    non_empty_node_cols = findall(nzs .> 1)
    for i ∈ non_empty_node_cols
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
        fig, ax = plt.subplots(1, 2, figsize=(4, 3.2), sharey=true)
        ax[1].plot(bx[i], z_cols[i])
        ax[1].plot(bxₕ[i], z_dg, "--")
        ax[1].set_xlabel(L"\partial_x b")
        ax[1].set_ylabel(L"z")
        ax[2].plot(by[i], z_cols[i])
        ax[2].plot(byₕ[i], z_dg, "--")
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
    i_p = 0
    for i ∈ non_empty_node_cols
        A = nuPGCM.get_baroclinic_LHS(z_cols[i], ε², f)
        r = nuPGCM.get_baroclinic_RHS(z_cols[i], bxₕ[i], byₕ[i], 0, 0, 0, 0, ε²)
        r_a = nuPGCM.get_baroclinic_RHS(z_cols[i], bx[i], by[i], 0, 0, 0, 0, ε²)
        sol = A\r
        nz = nzs[i]
        ωx[i_p+1:i_p+nz] = sol[0*nz+1:1*nz]
        ωy[i_p+1:i_p+nz] = sol[1*nz+1:2*nz]
        χx[i_p+1:i_p+nz] = sol[2*nz+1:3*nz]
        χy[i_p+1:i_p+nz] = sol[3*nz+1:4*nz]
        sol_a = A\r_a
        ωx_a[i_p+1:i_p+nz]= sol_a[0*nz+1:1*nz]
        ωy_a[i_p+1:i_p+nz]= sol_a[1*nz+1:2*nz]
        χx_a[i_p+1:i_p+nz]= sol_a[2*nz+1:3*nz]
        χy_a[i_p+1:i_p+nz]= sol_a[3*nz+1:4*nz]

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

        i_p += nz
    end
    ωx_e = maximum(abs.(ωx - ωx_a))/maximum(abs.(ωx_a))
    ωy_e = maximum(abs.(ωy - ωy_a))/maximum(abs.(ωy_a))
    χx_e = maximum(abs.(χx - χx_a))/maximum(abs.(χx_a))
    χy_e = maximum(abs.(χy - χy_a))/maximum(abs.(χy_a))
    println(@sprintf("%1.1e  %1.1e  %1.1e  %1.1e", ωx_e, ωy_e, χx_e, χy_e))
end

convergence_mixed(0.01)

println("Done.")