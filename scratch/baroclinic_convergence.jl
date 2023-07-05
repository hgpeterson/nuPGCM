using nuPGCM
using PyPlot
using Printf
using WriteVTK

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

p_ref = [ 0  0  0
          1  0  0
          0  1  0
          0  0  1
          1  0  1
          0  1  1]

function φ(ξ, i)
    if i == 1
        return (1 - ξ[1] - ξ[2])*(1 - ξ[3])
    elseif i == 2
        return ξ[1]*(1 - ξ[3])
    elseif i == 3
        return ξ[2]*(1 - ξ[3])
    elseif i == 4
        return (1 - ξ[1] - ξ[2])*ξ[3]
    elseif i == 5
        return ξ[1]*ξ[3]
    elseif i == 6
        return ξ[2]*ξ[3]
    end
end
function φξ(ξ, i)
    if i == 1
        return -(1 - ξ[3])
    elseif i == 2
        return (1 - ξ[3])
    elseif i == 3
        return 0
    elseif i == 4
        return -ξ[3]
    elseif i == 5
        return ξ[3]
    elseif i == 6
        return 0
    end
end
function φη(ξ, i)
    if i == 1
        return -(1 - ξ[3])
    elseif i == 2
        return 0
    elseif i == 3
        return (1 - ξ[3])
    elseif i == 4
        return -ξ[3]
    elseif i == 5
        return 0
    elseif i == 6
        return ξ[3]
    end
end
function φζ(ξ, i)
    if i == 1
        return -(1 - ξ[1] - ξ[2])
    elseif i == 2
        return -ξ[1]
    elseif i == 3
        return -ξ[2]
    elseif i == 4
        return (1 - ξ[1] - ξ[2])
    elseif i == 5
        return ξ[1]
    elseif i == 6
        return ξ[2]
    end
end

xξ(ξ, p) = sum(φξ(ξ, i)*p[i, 1] for i=1:6)
yξ(ξ, p) = sum(φξ(ξ, i)*p[i, 2] for i=1:6)
zξ(ξ, p) = sum(φξ(ξ, i)*p[i, 3] for i=1:6)
xη(ξ, p) = sum(φη(ξ, i)*p[i, 1] for i=1:6)
yη(ξ, p) = sum(φη(ξ, i)*p[i, 2] for i=1:6)
zη(ξ, p) = sum(φη(ξ, i)*p[i, 3] for i=1:6)
xζ(ξ, p) = sum(φζ(ξ, i)*p[i, 1] for i=1:6)
yζ(ξ, p) = sum(φζ(ξ, i)*p[i, 2] for i=1:6)
zζ(ξ, p) = sum(φζ(ξ, i)*p[i, 3] for i=1:6)
J(ξ, p) = inv([xξ(ξ, p) xη(ξ, p) xζ(ξ, p)
               yξ(ξ, p) yη(ξ, p) yζ(ξ, p)
               zξ(ξ, p) zη(ξ, p) zζ(ξ, p)])

function convergence_element_col(h)
    # params 
    ε² = 1e-2
    f = 1

    # surface triangle 
    p_sfc = [0      0
             2h/√3  0
             h/√3   h]

    # depths
    # H = [1, 1+h, 1]
    H = [1, 1, 1]

    # node grids
    z_cols = [0:-h:-H[i] for i=1:3]
    nzs = [size(z_cols[i], 1) for i=1:3]

    # 3d points
    np = sum(nzs)
    # nt = minimum(nzs)
    nt = minimum(nzs) - 1
    p = zeros(np, 3)
    t = Vector{Vector{Int64}}(undef, nt)
    for k=1:nt-1
        i = 3k-3
        for j=1:3
            p[i+j, 3] = z_cols[j][k]
            p[i+j, 1] = p_sfc[j, 1]
            p[i+j, 2] = p_sfc[j, 2]
            p[i+j+3, 3] = z_cols[j][k+1]
            p[i+j+3, 1] = p_sfc[j, 1]
            p[i+j+3, 2] = p_sfc[j, 2]
        end
        t[k] = collect(i+1:i+6)
        display(t)
    end
    # # last element is a tetrahedron!
    # p[np, 1:2] = p_sfc[1, :]
    # p[np, 3] = -1 - h
    # t[nt] = collect(np-3:np)

    vtk_grid("output/col.vtu", p', [MeshCell(size(t[k], 1) == 4 ? VTKCellTypes.VTK_TETRA : VTKCellTypes.VTK_WEDGE, t[k]) for k ∈ eachindex(t)]) do vtk end

    # buoyancy
    x = p[:, 1]
    y = p[:, 2]
    z = p[:, 3]
    b = @. exp(z)*sin(x)*cos(y)
    bx = [@.  exp(z_cols[i])*cos(p_sfc[i, 1])*cos(p_sfc[i, 2]) for i=1:3]
    by = [@. -exp(z_cols[i])*sin(p_sfc[i, 1])*sin(p_sfc[i, 2]) for i=1:3] 

    # numerical gradients
    bxₕ = [zeros(2nz-2) for nz ∈ nzs]
    byₕ = [zeros(2nz-2) for nz ∈ nzs]
    # prisms first
    jacobian = J(p[1, :], p[t[1], :]) # same for all elements in this case
    ξx = jacobian[1, 1]
    ηx = jacobian[2, 1]
    ζx = jacobian[3, 1]
    ξy = jacobian[1, 2]
    ηy = jacobian[2, 2]
    ζy = jacobian[3, 2]
    Dξ = [φξ(p_ref[i, :], j) for i=1:6, j=1:6]
    Dη = [φη(p_ref[i, :], j) for i=1:6, j=1:6]
    Dζ = [φζ(p_ref[i, :], j) for i=1:6, j=1:6]
    for i=1:3
        i1 = i 
        i2 = i + 3
        for k=1:nt-1
            bxₕ[i][2k-1] = sum(b[t[k][j]]*(Dξ[i1, j]*ξx + Dη[i1, j]*ηx + Dζ[i1, j]*ζx) for j=1:6)
            bxₕ[i][2k]   = sum(b[t[k][j]]*(Dξ[i2, j]*ξx + Dη[i2, j]*ηx + Dζ[i2, j]*ζx) for j=1:6)
            byₕ[i][2k-1] = sum(b[t[k][j]]*(Dξ[i1, j]*ξy + Dη[i1, j]*ηy + Dζ[i1, j]*ζy) for j=1:6)
            byₕ[i][2k]   = sum(b[t[k][j]]*(Dξ[i2, j]*ξy + Dη[i2, j]*ηy + Dζ[i2, j]*ζy) for j=1:6)
        end
    end
    # # tetrahedron
    # jacobian = Jacobians(p[t[end], :], [1 2 3 4]).Js[1, :, :]
    # ξx = jacobian[1, 1]
    # ηx = jacobian[2, 1]
    # ζx = jacobian[3, 1]
    # ξy = jacobian[1, 2]
    # ηy = jacobian[2, 2]
    # ζy = jacobian[3, 2]
    # sf = ShapeFunctions(order=1, dim=3)
    # p_ref_tet = nuPGCM.reference_element_nodes(1, 3)
    # Dξ = [∂φ∂ξ(sf, j, p_ref_tet[i, :]) for i=1:4, j=1:4]
    # Dη = [∂φ∂η(sf, j, p_ref_tet[i, :]) for i=1:4, j=1:4]
    # Dζ = [∂φ∂ζ(sf, j, p_ref_tet[i, :]) for i=1:4, j=1:4]
    # i = argmax(nzs) # column with extra node
    # k = nt 
    # i1 = i
    # i2 = 4
    # bxₕ[i][2k-1] = sum(b[t[k][j]]*(Dξ[i1, j]*ξx + Dη[i1, j]*ηx + Dζ[i1, j]*ζx) for j=1:4)
    # bxₕ[i][2k]   = sum(b[t[k][j]]*(Dξ[i2, j]*ξx + Dη[i2, j]*ηx + Dζ[i2, j]*ζx) for j=1:4)
    # byₕ[i][2k-1] = sum(b[t[k][j]]*(Dξ[i1, j]*ξy + Dη[i1, j]*ηy + Dζ[i1, j]*ζy) for j=1:4)
    # byₕ[i][2k]   = sum(b[t[k][j]]*(Dξ[i2, j]*ξy + Dη[i2, j]*ηy + Dζ[i2, j]*ζy) for j=1:4)

    for i=1:3
        z_dg = zeros(2nzs[i]-2)
        for j=1:nzs[i]-1
            z_dg[2j-1] = z_cols[i][j]
            z_dg[2j] = z_cols[i][j+1]
        end
        fig, ax = plt.subplots(1, figsize=(2, 3.2))
        ax.plot(bx[i], z_cols[i], "C0")
        ax.plot(bxₕ[i], z_dg, "C1--")
        ax.set_xlabel(L"\partial_x b")
        ax.set_ylabel(L"z")
        savefig("scratch/images/bx$i.png")
        println("scratch/images/bx$i.png")
        plt.close()
        fig, ax = plt.subplots(1, figsize=(2, 3.2))
        ax.plot(by[i], z_cols[i], "C0")
        ax.plot(byₕ[i], z_dg, "C1--")
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
    for i=1:3
        perm = sortperm(z_cols[i])
        A = nuPGCM.get_baroclinic_LHS(z_cols[i][perm], ε², f)
        r = nuPGCM.get_baroclinic_RHS(z_cols[i][perm], bxₕ[i][perm], byₕ[i][perm], 0, 0, 0, 0, ε²)
        r_a = nuPGCM.get_baroclinic_RHS(z_cols[i][perm], bx[i][perm], by[i][perm], 0, 0, 0, 0, ε²)
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

        plot(sol_a[0*nz+1:1*nz], z_cols[i][perm])
        plot(sol[0*nz+1:1*nz], z_cols[i][perm], "--")
        savefig("scratch/images/omegax$i.png")
        plt.close()
        println(maximum(abs.(sol - sol_a)))
    end
    ωx_e = maximum(abs.(ωx - ωx_a))
    ωy_e = maximum(abs.(ωy - ωy_a))
    χx_e = maximum(abs.(χx - χx_a))
    χy_e = maximum(abs.(χy - χy_a))
    println(@sprintf("%1.1e %1.1e %1.1e %1.1e", ωx_e, ωy_e, χx_e, χy_e))
end

function convergence_node_col()
    ε² = 1e-2
    f = 1
    ωx_a(z) = exp(z) - 1
    ωy_a(z) = -π/2*sin(π*z/2)
    χx_a(z) = -exp(z) + 1/2*(1 + z)^2 + (2 + z)/exp(1)
    χy_a(z) = -2/π*(1 + sin(π*z/2))
    bx(z) = 1 - exp(z) + ε²*π^3/8*sin(π*z/2)
    by(z) = -ε²*exp(z) + π/2*sin(π*z/2)
    for nz=[2^5, 2^6, 2^8, 2^9]
        z = -1:1/(nz-1):0
        z_half = (z[2:end] + z[1:end-1])/2
        z_dg = zeros(2nz-2)
        for i=1:nz-1
            z_dg[2i-1] = z[i]
            z_dg[2i] = z[i+1]
        end
        A = nuPGCM.get_baroclinic_LHS(z, ε², f)
        # r = nuPGCM.get_baroclinic_RHS(z, bx.(z), by.(z), -χy_a(0), χx_a(0), 0, 0, ε²)
        # r = nuPGCM.get_baroclinic_RHS(z, bx.(z_half), by.(z_half), -χy_a(0), χx_a(0), 0, 0, ε²)
        r = nuPGCM.get_baroclinic_RHS(z, bx.(z_dg), by.(z_dg), -χy_a(0), χx_a(0), 0, 0, ε²)
        sol = A\r
        ωx = sol[0*nz+1:1*nz]
        ωy = sol[1*nz+1:2*nz]
        χx = sol[2*nz+1:3*nz]
        χy = sol[3*nz+1:4*nz]
        ωx_e = maximum(abs.(ωx - ωx_a.(z)))
        ωy_e = maximum(abs.(ωy - ωy_a.(z)))
        χx_e = maximum(abs.(χx - χx_a.(z)))
        χy_e = maximum(abs.(χy - χy_a.(z)))
        println(@sprintf("%1.1e %1.1e %1.1e %1.1e", ωx_e, ωy_e, χx_e, χy_e))
        println(@sprintf("%1.1e %1.1e %1.1e %1.1e", ωx_e/4, ωy_e/4, χx_e/4, χy_e/4))
    end
end

convergence_element_col(0.1)

println("Done.")