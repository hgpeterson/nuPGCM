using nuPGCM
using PyPlot
using Printf
using Delaunay

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("output")

function convergence_element_col(h)
    # params 
    ε² = 1e-2
    f = 1

    # surface triangle 
    p_sfc = [0      0
             2h/√3  0
             h/√3   h]

    # node grid
    z = -1:h:0 
    nz = size(z, 1) 

    # 3d points
    p = zeros(3nz, 3)
    for i=1:3
        p[(i-1)*nz+1:i*nz, 1:2] = repeat(p_sfc[i, :]', nz, 1)
        p[(i-1)*nz+1:i*nz, 3] = z
    end

    # edges
    e = Dict("sfc"=>[1, nz+1, 2nz+1], "bot"=>[nz, 2nz, 3nz])

    # tessellate
    t = Matrix{Int64}(undef, 0, 4) 
    top = [1, 1+nz, 1+2nz]
    for j=2:nz
        bot = top .+ 1
        ig = unique(vcat(top, bot))
        tl = delaunay(p[ig, :]).simplices
        t = [t; ig[tl]]
        top = bot
    end

    # element column
    g = Grid(1, p, t, e)
    gb = Grid(1, p, t, e)

    # for b gradients
    Dxs, Dys = nuPGCM.get_b_gradient_matrices(gb, g, [nz, nz, nz]) 

    # buoyancy
    b = FEField(x->exp(x[3])*sin(x[1])*cos(x[2]), gb)
    bx(x) = exp(x[3])*cos(x[1])*cos(x[2])
    by(x) = -exp(x[3])*sin(x[1])*sin(x[2])

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
        r = nuPGCM.get_baroclinic_RHS(z, Dxs[i]*b.values, Dys[i]*b.values, 0, 0, 0, 0, ε²)
        bx_a(z) = bx([p_sfc[i, 1], p_sfc[i, 2], z])
        by_a(z) = by([p_sfc[i, 1], p_sfc[i, 2], z])
        r_a = nuPGCM.get_baroclinic_RHS(z, bx_a.(z), by_a.(z), 0, 0, 0, 0, ε²)
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

convergence_element_col(0.01)

println("Done.")