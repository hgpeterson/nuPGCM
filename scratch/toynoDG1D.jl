using nuPGCM
using WriteVTK
using PyPlot
using SparseArrays
using LinearAlgebra

include("utils.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
Solve
    -∂zz(u) + u = ∂x(b),
with
    • u = 0 at z = 0,
    • u = 0 or ∫ zu dz = 0 at z = -H
"""
function solve_toynoDG1D()
    x = g.p[:, 1]
    z = g.p[:, 2]
    N = size(g.p, 1)

    # for element matricies
    s = ShapeFunctionIntegrals(g.s, g.s)
    J = Jacobians(g)

    # separate mesh into columns
    cols = get_cols(g.p, g.t)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k ∈ axes(g.t, 1)
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        Cx = J.dets[k]*sum(s.C.*J.Js[k, :, 1], dims=1)[1, :, :]
        M = J.dets[k]*s.M
        rᵏ = Cx*b.(x[g.t[k, :]], z[g.t[k, :]])

        # interior terms
        for i=1:g.nn
            if g.t[k, i] ∉ g.e
                # ∫_Ω ∂x(b)φ dxdz
                r[g.t[k, i]] += rᵏ[i]
                for j=1:g.nn
                    # ∫_Ω ∂z(φᵢ)∂z(φⱼ) dxdz
                    push!(A, (g.t[k, i], g.t[k, j], K[i, j]))
                    # ∫_Ω φᵢφⱼ dxdz
                    push!(A, (g.t[k, i], g.t[k, j], M[i, j]))
                end
            end
        end

    end

    # dirichlet
    bot, sfc = get_sides(g)
    for i ∈ sfc
        push!(A, (i, i, 1))
    end
    for i ∈ bot
        # u = 0
        push!(A, (i, i, 1))

        # # ∫ zu dz = 0
        # for j=1:nz-1
        #     push!(A, (ωymap[1], ωymap[j],     z[j]*(z[j+1] - z[j])/2))
        #     push!(A, (ωymap[1], ωymap[j+1], z[j+1]*(z[j+1] - z[j])/2))
        # end
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # solve
    u = A\r

    # exact solution
    u_a = @. -(bx(x, z)*exp(-z)*(-1 + exp(z))*(-1 + exp(H(x, z) + z)))/(1 + exp(H(x, z)))
    # u_a = @. E^-z (-1 + E^z) (bx - (bx E^H (-1 + E^H - H))/(-1 + E^H)^2 - (bx E^(H + z) (-1 + E^H - H))/(-1 + E^H)^2)

    # save as .vtu
    points = g.p'
    cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, g.t[i, :]) for i ∈ axes(g.t, 1)]
    vtk_grid("output/u.vtu", points, cells) do vtk
        vtk["u"] = u
        vtk["uₐ"] = u_a
    end
end

function get_cols(p, t)
    # number of columns is half the number of boundary edges
    emap, edges, bndix = all_edges(t)
    bnd_edges = edges[bndix, :]
    ncols = Int64(1/2*size(bnd_edges, 1))

    # bounds to each column
    x = range(-1, 1, length=ncols+1)

    # k = cols[i][j] = jᵗʰ element of iᵗʰ col
    cols = [Int64[] for i=1:ncols]

    for k ∈ axes(t, 1)
        # find which column centroid in x lives in
        x̄ = sum(p[t[k, :], 1])/3
        i = searchsorted(x, x̄).stop
        push!(cols[i], k)
    end

    return cols
end

g = FEGrid("meshes/valign2D/mesh2.h5", 1)
b(x, z) = x
bx(x, z) = 1
H(x, z) = 1 - x^2
solve_toynoDG1D()