## Solve
##     -ε²∂zz(ωˣ) - ωʸ =  ∂y(b),
##     -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
##        ∂zz(χˣ) + ωˣ = 0,
##        ∂zz(χʸ) + ωʸ = 0,
## with bc
## At z = 0:
##     • ωˣ = 0, ωʸ = 0, χˣ = Uʸ, χʸ = -Uˣ
## At z = -H:
##     • χˣ = 0, χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0

using nuPGCM
using WriteVTK
using HDF5
using Delaunay
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function gen_mesh(ifile; order)
    # load mesh of circle
    file = h5open(ifile, "r")
    p_sfc = read(file, "p")
    t_sfc = Int64.(read(file, "t"))
    e_sfc = Int64.(read(file, "e")[:, 1])
    close(file)
    x = p_sfc[:, 1]
    y = p_sfc[:, 2]
    np_sfc = size(p_sfc, 1)
    nt_sfc = size(t_sfc, 1)

    # mesh res
    emap, edges, bndix = all_edges(t_sfc)
    h = 1/size(edges, 1)*sum(norm(p_sfc[edges[i, 1], :] - p_sfc[edges[i, 2], :]) for i in axes(edges, 1))

    # mapping from points to triangles:
    #   `p_to_tri[i]` is vector of cartesian indices pointing to where point `i` is in `t_sfc`
    p_to_tri = [findall(I -> i ∈ t_sfc[I], CartesianIndices(size(t_sfc))) for i=1:np_sfc]

    # mapping from triangles to points in 3D: 
    #   `tri_to_p[k, i][j]` is the `j`th point in the vertical for the `i`th point of triangle `k`
    tri_to_p = [Int64[] for k=1:nt_sfc, i=1:3] # allocate

    # add points to p, e, and tri_to_p
    nzs = Int64[i ∈ e_sfc ? 1 : ceil(H(x[i], y[i])/h) for i=1:np_sfc]
    p = zeros(sum(nzs), 3)
    e = Dict("sfc"=>Int64[], "bot"=>Int64[])
    np = 0
    for i=1:np_sfc
        # vertical grid
        nz = nzs[i]
        if nz == 1
            z = [0]
        else
            z = -range(0, H(x[i], y[i]), length=nz)
        end

        # add to p
        p[np+1:np+nz, :] = [x[i]*ones(nz)  y[i]*ones(nz)  z]

        # add to e
        e["sfc"] = [e["sfc"]; np + 1]
        if nz != 1
            e["bot"] = [e["sfc"]; np + nz]
        end

        # add to tri_to_p
        for I ∈ p_to_tri[i]
            for j=np+1:np+nz
                push!(tri_to_p[I], j)
            end
        end

        # iterate
        np += nz
    end

    # setup shape functions and their integrals now since they're the same for each grid
    sf = ShapeFunctions(order=order, dim=3)
    sfi = ShapeFunctionIntegrals(sf, sf)

    # columnwise and global tessellation
    cols = Vector{FEGrid}(undef, nt_sfc)
    t = Matrix{Int64}(undef, 0, 4) 
    for k=1:nt_sfc
        # number of points in vertical for each vertex of sfc tri
        lens = length.(tri_to_p[k, :])

        # local p and e for column
        nodes_col = [tri_to_p[k, 1]; tri_to_p[k, 2]; tri_to_p[k, 3]]
        p_col = p[nodes_col, :]  
        e_sfc_col = [1, lens[1]+1, lens[1]+lens[2]+1]
        e_bot_col = [lens[1], lens[1]+lens[2], lens[1]+lens[2]+lens[3]]

        # start local t
        t_col = Matrix{Int64}(undef, 0, 4) 

        # first top tri is at sfc
        top = [tri_to_p[k, i][1] for i=1:3]

        # continue down to bottom
        for j=2:maximum(lens)
            # make bottom tri from next nodes down or top tri nodes
            bot = [j ≤ lens[i] ? tri_to_p[k, i][j] : top[i] for i=1:3]

            # use delaunay to tessellate
            ig = unique(vcat(top, bot))
            tl = delaunay(p[ig, :]).simplices

            # add to t
            t = [t; ig[tl]]

            # add to t_col
            i_col = Int64.(indexin(ig, nodes_col))
            t_col = [t_col; i_col[tl]]

            # continue
            top = bot
        end

        # create e_col dictionary
        e_col = Dict("sfc"=>e_sfc_col, "bot"=>e_bot_col)

        # save column data
        cols[k] = FEGrid(order, p_col, t_col, e_col, sf, sfi)

        # remove from bot if in sfc
        cols[k].e["bot"] = cols[k].e["bot"][findall(i -> cols[k].e["bot"][i] ∉ cols[k].e["sfc"], 1:size(cols[k].e["bot"], 1))]
    end

    g = FEGrid(order, p, t, e)

    return cols, g
end

function var_indices(col)
    ωxmap = 0*col.np+1:1*col.np
    ωymap = 1*col.np+1:2*col.np
    χxmap = 2*col.np+1:3*col.np
    χymap = 3*col.np+1:4*col.np
    return ωxmap, ωymap, χxmap, χymap
end

function get_sol(col, b, Ux, Uy)
    # indices
    ωxmap, ωymap, χxmap, χymap = var_indices(col)
    N = 4*col.np

    # unpack
    sfc = col.e["sfc"]
    bot = col.e["bot"]
    J = col.J
    s = col.sfi

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k=1:col.nt
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M
        Cx = J.dets[k]*sum(s.C.*J.Js[k, :, 1], dims=1)[1, :, :]
        Cy = J.dets[k]*sum(s.C.*J.Js[k, :, 2], dims=1)[1, :, :]

        # interior terms
        for i=1:col.nn, j=1:col.nn
            # indices
            ωxi = ωxmap[col.t[k, :]]
            ωyi = ωymap[col.t[k, :]]
            χxi = χxmap[col.t[k, :]]
            χyi = χymap[col.t[k, :]]
            if col.t[k, i] ∉ sfc && col.t[k, i] ∉ bot
                # eq 1: ε²∂z(ωˣ)∂z(ωˣ)
                push!(A, (ωxi[i], ωxi[j], ε²*K[i, j]))
                # eq 1: -ωʸωˣ
                push!(A, (ωxi[i], ωyi[j], -M[i, j]))

                # eq 2: ε²∂z(ωʸ)∂z(ωʸ)
                push!(A, (ωyi[i], ωyi[j], ε²*K[i, j]))
                # eq 2: ωˣωʸ
                push!(A, (ωyi[i], ωxi[j],  M[i, j]))
            end
            if col.t[k, i] ∉ sfc
                # eq 3: -∂z(χˣ)∂z(χˣ)
                push!(A, (χxi[i], χxi[j], -K[i, j]))
                # eq 3: ωˣχˣ
                push!(A, (χxi[i], ωxi[j],  M[i, j]))

                # eq 4: ∂z(χʸ)∂z(χʸ)
                push!(A, (χyi[i], χyi[j], -K[i, j]))
                # eq 4: ωʸχʸ
                push!(A, (χyi[i], ωyi[j],  M[i, j]))
            end
        end

        # ∂y(b)ωˣ
        r[ωxmap[col.t[k, :]]] += Cy*b[col.t[k, :]]
        # -∂x(b)ωʸ
        r[ωymap[col.t[k, :]]] -= Cx*b[col.t[k, :]]
        # p_tet = col.p[col.t[k, :], :]
        # x = p_tet[:, 1]
        # y = p_tet[:, 2]
        # z = p_tet[:, 3]
        # r[ωxmap[col.t[k, :]]] += M*f1.(x, y, z)
        # r[ωymap[col.t[k, :]]] += M*f2.(x, y, z)
        # r[χxmap[col.t[k, :]]] += M*f3.(x, y, z)
        # r[χymap[col.t[k, :]]] += M*f4.(x, y, z)
    end

    # surface nodes 
    for i ∈ sfc
        push!(A, (ωxmap[i], ωxmap[i], 1))
        push!(A, (ωymap[i], ωymap[i], 1))
        push!(A, (χxmap[i], χxmap[i], 1))
        push!(A, (χymap[i], χymap[i], 1))
        r[ωxmap[i]] = 0
        r[ωymap[i]] = 0
        r[χxmap[i]] = Uy[i]
        r[χymap[i]] = -Ux[i]
        # x = col.p[i, 1]
        # y = col.p[i, 2]
        # r[ωxmap[i]] = ωx_a(x, y, 0)
        # r[ωymap[i]] = ωy_a(x, y, 0)
        # r[χxmap[i]] = χx_a(x, y, 0)
        # r[χymap[i]] = χy_a(x, y, 0)
    end

    # bottom nodes
    for i ∈ bot
        push!(A, (ωxmap[i], χxmap[i], 1))
        push!(A, (ωymap[i], χymap[i], 1))
        # push!(A, (ωxmap[i], ωxmap[i], 1))
        # push!(A, (ωymap[i], ωymap[i], 1))
        # push!(A, (χxmap[i], χxmap[i], 1))
        # push!(A, (χymap[i], χymap[i], 1))
        r[ωxmap[i]] = 0
        r[ωymap[i]] = 0
        # x = col.p[i, 1]
        # y = col.p[i, 2]
        # r[ωxmap[i]] = χx_a(x, y, -H(x, y))
        # r[ωymap[i]] = χy_a(x, y, -H(x, y))
        # r[ωxmap[i]] = ωx_a(x, y, -H(x, y))
        # r[ωymap[i]] = ωy_a(x, y, -H(x, y))
        # r[χxmap[i]] = χx_a(x, y, -H(x, y))
        # r[χymap[i]] = χy_a(x, y, -H(x, y))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # solve
    return A\r
end

function fd_sol(z, bx, by, ε², Ux, Uy)
    # indices
    nz = size(z, 1)
    ωxmap = 1:nz
    ωymap = (nz+1):2*nz

    # matrix
    A = Tuple{Int64,Int64,Float64}[]  
    r = zeros(2*nz)

    # interior nodes
    for j=2:nz-1 
        # ∂zz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # eqtn 1: -ε²∂zz(ωˣ) - ωʸ = ∂y(b)
        # term 1
        push!(A, (ωxmap[j], ωxmap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωxmap[j], ωxmap[j],   -ε²*fd_zz[2]))
        push!(A, (ωxmap[j], ωxmap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωxmap[j], ωymap[j], -1))
        # rhs
        r[ωxmap[j]] = by[j]

        # eqtn 2: -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
        # term 1
        push!(A, (ωymap[j], ωymap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωymap[j], ωymap[j],   -ε²*fd_zz[2]))
        push!(A, (ωymap[j], ωymap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωymap[j], ωxmap[j], 1))
        # rhs
        r[ωymap[j]] = -bx[j]
    end

    # ωˣ = ωʸ = 0 at z = 0
    push!(A, (ωxmap[nz], ωxmap[nz], 1))
    push!(A, (ωymap[nz], ωymap[nz], 1))

    # ∫ zωˣ dz = Uy
    for j=1:nz-1
        push!(A, (ωxmap[1], ωxmap[j],     z[j]*(z[j+1] - z[j])/2))
        push!(A, (ωxmap[1], ωxmap[j+1], z[j+1]*(z[j+1] - z[j])/2))
    end
    r[ωxmap[1]] = Uy

    # ∫ zωʸ dz = -Ux
    for j=1:nz-1
        push!(A, (ωymap[1], ωymap[j],     z[j]*(z[j+1] - z[j])/2))
        push!(A, (ωymap[1], ωymap[j+1], z[j+1]*(z[j+1] - z[j])/2))
    end
    r[ωymap[1]] = -Ux

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), 2*nz, 2*nz)

    sol = A\r
    return sol[ωxmap], sol[ωymap]
end

function plot_1D(col, sol)
    # indices
    ωxmap, ωymap, χxmap, χymap = var_indices(col)

    # unpack
    ωx = FEField(sol[ωxmap], col)
    ωy = FEField(sol[ωymap], col)
    χx = FEField(sol[χxmap], col)
    χy = FEField(sol[χymap], col)

    # compare with high res FD solution
    x = 1/size(col.e["sfc"],1)*sum(col.p[col.e["sfc"][:], 1])
    y = 1/size(col.e["sfc"],1)*sum(col.p[col.e["sfc"][:], 2])
    z = -H(x, y):H(x, y)/2^10:0
    ωx_fd, ωy_fd = fd_sol(z, bx.(x, y, z), by.(x, y, z), ε², Ux(x, y), Uy(x, y))
    ωx_f(z) = evaluate(ωx, [x, y, z])
    ωy_f(z) = evaluate(ωy, [x, y, z])
    χx_f(z) = evaluate(χx, [x, y, z])
    χy_f(z) = evaluate(χy, [x, y, z])
    println(@sprintf("Max error ωx: %1.1e", maximum(x->isnan(x) ? -Inf : x, abs.(ωx_f.(z) - ωx_fd))))
    println(@sprintf("Max error ωy: %1.1e", maximum(x->isnan(x) ? -Inf : x, abs.(ωy_f.(z) - ωy_fd))))

    # plot
    fig, ax = subplots(1, 2, figsize=(2*2, 3.2), sharey=true)
    ax[1].plot(ωx_f.(z), z, label=L"\omega^x")
    ax[1].plot(ωy_f.(z), z, label=L"\omega^y")
    ax[1].plot(ωx_fd, z, "k--", lw=0.5, label="“Truth”")
    ax[1].plot(ωy_fd, z, "k--", lw=0.5)
    ax[2].plot(χx_f.(z), z, label=L"\chi^x")
    ax[2].plot(χy_f.(z), z, label=L"\chi^y")
    ax[1].legend()
    ax[2].legend()
    ax[1].set_xlabel(L"\omega")
    ax[1].set_ylabel(L"z")
    ax[2].set_xlabel(L"\chi")
    savefig("scratch/images/omega_chi.png")
    println("scratch/images/omega_chi.png")
    plt.close()
end

function plot_3D()
    # global p, t, e
    np = sum(col.np for col ∈ cols)
    nt = sum(col.nt for col ∈ cols)
    nsfc = sum(size(col.e["sfc"], 1) for col ∈ cols) 
    nbot = sum(size(col.e["bot"], 1) for col ∈ cols) 
    p = zeros(Float64, (np, 3))
    t = zeros(Int64, (nt, cols[1].nn))
    sfc = zeros(Int64, (nsfc,))
    bot = zeros(Int64, (nbot,))

    # global solutions
    ωx = zeros(np)
    ωy = zeros(np)
    χx = zeros(np)
    χy = zeros(np)

    # current indices
    i_p = 0
    i_t = 0
    i_sfc = 0
    i_bot = 0

    # all the nodes within each column will have a unique tag
    for i ∈ eachindex(cols)
        # column
        col = cols[i]
        nsfc_col = size(col.e["sfc"], 1)
        nbot_col = size(col.e["bot"], 1)

        # add nodes, triangles, and edge nodes
        p[i_p+1:i_p+col.np, :] = col.p
        t[i_t+1:i_t+col.nt, :] = i_p .+ col.t
        sfc[i_sfc+1:i_sfc+nsfc_col] = i_p .+ col.e["sfc"]
        bot[i_bot+1:i_bot+nbot_col] = i_p .+ col.e["bot"]

        # unpack solutions
        ωx[i_p+1:i_p+col.np] = sols[i][0*col.np+1:1*col.np]
        ωy[i_p+1:i_p+col.np] = sols[i][1*col.np+1:2*col.np]
        χx[i_p+1:i_p+col.np] = sols[i][2*col.np+1:3*col.np]
        χy[i_p+1:i_p+col.np] = sols[i][3*col.np+1:4*col.np]

        # increment
        i_p += col.np
        i_t += col.nt
        i_sfc += nsfc_col
        i_bot += nbot_col
    end

    # err_ωx = abs.(ωx - ωx_a.(p[:, 1], p[:, 2], p[:, 3]))
    # err_ωy = abs.(ωy - ωy_a.(p[:, 1], p[:, 2], p[:, 3]))
    # err_χx = abs.(χx - χx_a.(p[:, 1], p[:, 2], p[:, 3]))
    # err_χy = abs.(χy - χy_a.(p[:, 1], p[:, 2], p[:, 3]))
    # println(@sprintf("Error ωx: %1.1e", maximum(err_ωx)))
    # println(@sprintf("Error ωy: %1.1e", maximum(err_ωy)))
    # println(@sprintf("Error χx: %1.1e", maximum(err_χx)))
    # println(@sprintf("Error χy: %1.1e", maximum(err_χy)))

    # save as .vtu
    if cols[1].order == 1
        cell_type = VTKCellTypes.VTK_TETRA
    elseif cols[1].order == 2
        cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
    end
    cells = [MeshCell(cell_type, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid("output/pg_vort_DG_3D.vtu", p', cells) do vtk
        vtk["ωx"] = ωx
        vtk["ωy"] = ωy
        vtk["χx"] = χx
        vtk["χy"] = χy
        # vtk["ωx_a"] = ωx_a.(p[:, 1], p[:, 2], p[:, 3])
        # vtk["ωy_a"] = ωy_a.(p[:, 1], p[:, 2], p[:, 3])
        # vtk["χx_a"] = χx_a.(p[:, 1], p[:, 2], p[:, 3])
        # vtk["χy_a"] = χy_a.(p[:, 1], p[:, 2], p[:, 3])

        bdy = zeros(np)
        bdy[sfc] .= 1
        vtk["sfc"] = bdy

        bdy = zeros(np)
        bdy[bot] .= 1
        vtk["bot"] = bdy
    end
    println("output/pg_vort_DG_3D.vtu")
end

# params
ε² = 1
Ux(x, y) = 0
Uy(x, y) = 0
# b(x, y, z) = x
# bx(x, y, z) = 1
# by(x, y, z) = 0
δ = 0.1
b(x, y, z) = z + δ*exp(-(z + H(x, y))/δ)
bx(x, y, z) = -Hx(x, y)*exp(-(z + H(x, y))/δ)
by(x, y, z) = -Hy(x, y)*exp(-(z + H(x, y))/δ)
H(x, y) = 1 - x^2 - y^2
Hx(x, y) = -2*x
Hy(x, y) = -2*y

# grid
cols, g = gen_mesh("meshes/circle/mesh2.h5", order=2)
println("ncols = ", size(cols, 1))

# b, Ux, Uy in each column
b_cols = [b.(col.p[:, 1], col.p[:, 2], col.p[:, 3]) for col ∈ cols]
Ux_cols = [Ux.(col.p[:, 1], col.p[:, 2]) for col ∈ cols]
Uy_cols = [Uy.(col.p[:, 1], col.p[:, 2]) for col ∈ cols]

# # constructed solution and forcing
# ωx_a(x, y, z) = x*z*exp(x*y*z)
# ωy_a(x, y, z) = y*z*exp(x*y*z)
# χx_a(x, y, z) = -(1 - H(x, y) + exp(z)*(-1 + H(x, y) + z))*cos(y)*sin(x)
# χy_a(x, y, z) = -(1 - H(x, y) + exp(z)*(-1 + H(x, y) + z))*cos(x)*sin(y)
# f1(x, y, z) = -y*exp(x*y*z)*(z + 2*x^2*ε² + x^3*y*z*ε²)
# f2(x, y, z) = -x*exp(x*y*z)*(2*y^2*ε² + z*(-1 + x*y^3*ε²))
# f3(x, y, z) = x*z*exp(x*y*z) - exp(z)*(1 + H(x, y) + z)*cos(y)*sin(x)
# f4(x, y, z) = y*z*exp(x*y*z) - exp(z)*(1 + H(x, y) + z)*cos(x)*sin(y)

# solve
sols = [get_sol(cols[i], b_cols[i], Ux_cols[i], Uy_cols[i]) for i ∈ eachindex(cols)]
sols = []
@showprogress "Solving..." for i ∈ eachindex(cols)
    push!(sols, get_sol(cols[i], b_cols[i], Ux_cols[i], Uy_cols[i]))
end

i_col = argmax([col.nt for col ∈ cols])
plot_1D(cols[i_col], sols[i_col])

plot_3D()

# Notes:
# dirichlet: O(h), O(h^2)
# neumann + dirichlet: O(h), O(h^2) for χ
# fd: O(h^2), ~O(h^3) for tallest column

println("Done.")