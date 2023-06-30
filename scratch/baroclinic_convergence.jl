using nuPGCM
using PyPlot
using Printf
using Delaunay

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
    p_sfc = [0      h
             2h/√3  h
             h/√3   2h]

    # node grid
    z = -1:h:0 
    nz = size(z, 1) 

    # 3d points
    p = zeros(3nz ,3)
    t = zeros(Int64, nz-1, 6)
    for k=1:nz-1
        i = 3k-2
        p[i:i+2, 3] .= z[k]
        p[i:i+2, 1] = p_sfc[:, 1]
        p[i:i+2, 2] = p_sfc[:, 2]
        p[i+3:i+5, 3] .= z[k+1]
        p[i+3:i+5, 1] = p_sfc[:, 1]
        p[i+3:i+5, 2] = p_sfc[:, 2]
        t[k, :] = i:i+5
    end
    e = [1, 2, 3, 3nz-2, 3nz-1, 3nz]

    # buoyancy
    x = p[:, 1]
    y = p[:, 2]
    z = p[:, 3]
    b = @. exp(z)*sin(x)*cos(y)
    bx = @.  exp(z)*cos(x)*cos(y)
    by = @. -exp(z)*sin(x)*sin(y)

    # numerical gradients
    bxₕ = zeros(3, 2nz-2)
    byₕ = zeros(3, 2nz-2)
    jacobian = J(p[1, :], p[t[1, :], :]) # same for all elements in this case
    ξx = jacobian[1, 1]
    ηx = jacobian[2, 1]
    ζx = jacobian[3, 1]
    ξy = jacobian[1, 2]
    ηy = jacobian[2, 2]
    ζy = jacobian[3, 2]
    Dξ = [φξ(p_ref[i, :], j) for i=1:6, j=1:6]
    Dη = [φη(p_ref[i, :], j) for i=1:6, j=1:6]
    Dζ = [φζ(p_ref[i, :], j) for i=1:6, j=1:6]
    for k=1:nz-1
        for i=1:3
            i1 = i 
            i2 = i + 3
            bxₕ[i, 2k-1] = sum(b[t[k, j]]*(Dξ[i1, j]*ξx + Dη[i1, j]*ηx + Dζ[i1, j]*ζx) for j=1:6)
            bxₕ[i, 2k]   = sum(b[t[k, j]]*(Dξ[i2, j]*ξx + Dη[i2, j]*ηx + Dζ[i2, j]*ζx) for j=1:6)
            byₕ[i, 2k-1] = sum(b[t[k, j]]*(Dξ[i1, j]*ξy + Dη[i1, j]*ηy + Dζ[i1, j]*ζy) for j=1:6)
            byₕ[i, 2k]   = sum(b[t[k, j]]*(Dξ[i2, j]*ξy + Dη[i2, j]*ηy + Dζ[i2, j]*ζy) for j=1:6)
        end
    end

    z = -1:h:0
    # z_dg = zeros(2nz-2)
    # for i=1:nz-1
    #     z_dg[2i-1] = z[i]
    #     z_dg[2i] = z[i+1]
    # end
    # fig, ax = plt.subplots(1, figsize=(2, 3.2))
    # # ax.plot(bx[1:3:end], z, "C0")
    # ax.plot(bxₕ[1, :], z_dg, "C1--")
    # # ax.plot(bx[2:3:end], z, "C2")
    # ax.plot(bxₕ[2, :], z_dg, "C3--")
    # # ax.plot(bx[3:3:end], z, "C4")
    # ax.plot(bxₕ[3, :], z_dg, "C5--")
    # ax.set_xlabel(L"\partial_x b")
    # ax.set_ylabel(L"z")
    # savefig("scratch/images/bx.png")
    # println("scratch/images/bx.png")
    # plt.close()

    # solve
    ωx = zeros(3nz)
    ωy = zeros(3nz)
    χx = zeros(3nz)
    χy = zeros(3nz)
    ωx_a = zeros(3nz)
    ωy_a = zeros(3nz)
    χx_a = zeros(3nz)
    χy_a = zeros(3nz)
    A = nuPGCM.get_baroclinic_LHS(z, ε², f)
    for i=1:3
        r = nuPGCM.get_baroclinic_RHS(z, bxₕ[i, :], byₕ[i, :], 0, 0, 0, 0, ε²)
        r_a = nuPGCM.get_baroclinic_RHS(z, bx[i:3:end], by[i:3:end], 0, 0, 0, 0, ε²)
        sol = A\r
        ωx[(i-1)*nz+1:i*nz] = sol[0*nz+1:1*nz]
        ωy[(i-1)*nz+1:i*nz] = sol[1*nz+1:2*nz]
        χx[(i-1)*nz+1:i*nz] = sol[2*nz+1:3*nz]
        χy[(i-1)*nz+1:i*nz] = sol[3*nz+1:4*nz]
        sol = A\r_a
        ωx_a[(i-1)*nz+1:i*nz]= sol[0*nz+1:1*nz]
        ωy_a[(i-1)*nz+1:i*nz]= sol[1*nz+1:2*nz]
        χx_a[(i-1)*nz+1:i*nz]= sol[2*nz+1:3*nz]
        χy_a[(i-1)*nz+1:i*nz]= sol[3*nz+1:4*nz]
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
        println(@sprintf("%1.1e %1.1e %1.1e %1.1e", ωx_e, ωy_e/4, χx_e/4, χy_e/4))
    end
end

convergence_element_col(0.04)

println("Done.")