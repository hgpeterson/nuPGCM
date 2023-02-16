using nuPGCM
using WriteVTK
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)


"""
Solve
    -ε²∂zz(ωˣ) - ωʸ = 0,
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b),
       ∂zz(χˣ) + ωˣ = 0,
       ∂zz(χʸ) + ωʸ = 0,
with bc
At z = 0:
    • ωˣ = 0, ωʸ = 0, χˣ = 0, χʸ = -Uˣ
At z = -H:
    • ωˣ = Uˣ/ε², χʸ = 0, ∂z(χˣ) = 0, ∂z(χʸ) = 0
"""
function get_LHS(col)
    # indices
    ωxmap = 1:col.np
    ωymap = (col.np+1):2*col.np
    χxmap = (2*col.np+1):3*col.np
    χymap = (3*col.np+1):4*col.np
    N = 4*col.np

    # sfc and bot
    sfc = col.e[col.p[col.e, 2] .== 0.0]
    bot = col.e[col.p[col.e, 2] .!= 0.0]

    # for element matricies
    J = Jacobians(col)
    s = ShapeFunctionIntegrals(col.s, col.s)

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    for k=1:col.nt
        # matrices
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M

        # interior terms
        for i=1:col.nn, j=1:col.nn
            # indices
            ωxi = ωxmap[col.t[k, :]]
            ωyi = ωymap[col.t[k, :]]
            χxi = χxmap[col.t[k, :]]
            χyi = χymap[col.t[k, :]]
            if col.t[k, i] ∉ col.e
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
    end

    # surface nodes 
    for i ∈ sfc
        push!(A, (ωxmap[i], ωxmap[i], 1))
        push!(A, (ωymap[i], ωymap[i], 1))
        push!(A, (χxmap[i], χxmap[i], 1))
        push!(A, (χymap[i], χymap[i], 1))
    end

    # bottom nodes
    for i ∈ bot
        push!(A, (ωxmap[i], ωxmap[i], 1))
        push!(A, (ωymap[i], χymap[i], 1))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)
    return lu(A)
end

function get_RHS(col)
    # indices
    ωxmap = 1:col.np
    ωymap = (col.np+1):2*col.np
    χxmap = (2*col.np+1):3*col.np
    χymap = (3*col.np+1):4*col.np
    N = 4*col.np

    # for element matricies
    s = ShapeFunctionIntegrals(col.s, col.s)
    J = Jacobians(col)

    # stamp
    r = zeros(N)
    for k=1:col.nt
        # triangle nodes
        p_tri = col.p[col.t[k, 1:3], :]

        # matrices
        Cx = J.dets[k]*sum(s.CT.*J.Js[k, :, 1], dims=1)[1, :, :]

        # 1D quadrature
        w, ξ = quad_weights_points(deg=2, dim=1)

        # 1D shape functions
        s1D = ShapeFunctions(order=1, dim=1)

        # which edge is vertical edge
        ie, edge = vert_edge(p_tri)

        # is it on left or right? +1 if right, -1 if left
        sign_multiplier = side_of_vert_edge(p_tri, ie)

        # z-coords of edge nodes
        z1 = p_tri[edge[1], 2]
        z2 = p_tri[edge[2], 2]

        # # connectivity pair for this edge
        # pair = connectivities[k, ie] # findall(I -> emap[I] == emap[k, ie] && I != CartesianIndex(k, ie), CartesianIndices(emap))[1]
        # k_pair = pair[1]
        # ie_pair = pair[2]
        # edge_pair = [ie_pair, mod1(ie_pair+1, 3)]
        # if g.p[g.t[k_pair, edge_pair]] != g.p[g.t[k, edge]]
        #     edge_pair = [mod1(ie_pair+1, 3), ie_pair]
        # end

        # average b values
        # b1 = (b_dg[t_dg[k, edge[1]]] + b_dg[t_dg[k_pair, edge_pair[1]]])/2
        # b2 = (b_dg[t_dg[k, edge[2]]] + b_dg[t_dg[k_pair, edge_pair[2]]])/2
        b1 = b(p_tri[edge[1], 1], p_tri[edge[1], 2])
        b2 = b(p_tri[edge[2], 1], p_tri[edge[2], 2])

        # ∫_∂Ωᵢₑ bφᵢ dz 
        for i=1:2
            f(t) = (b1*φ(s1D, 1, t) + b2*φ(s1D, 2, t))*φ(s1D, i, t)
            ∫f = dot(w, f.(ξ))*abs(z2 - z1)/2
            r[ωymap[col.t[k, edge[i]]]] -= sign_multiplier*∫f
        end

        # -∫_Ω b∂x(φ) dxdz
        r[ωymap[col.t[k, :]]] += Cx*b(p_tri[:, 1], p_tri[:, 2])
    end

    # surface nodes 
    sfc = col.e[col.p[col.e, 2] .== 0.0]
    for i ∈ sfc
        r[ωxmap[i]] = 0
        r[ωymap[i]] = 0
        r[χxmap[i]] = 0
        r[χymap[i]] = -Ux
    end

    # bottom nodes
    bot = col.e[col.p[col.e, 2] .!= 0.0]
    for i ∈ bot
        r[ωxmap[i]] = Ux/ε²
        r[ωymap[i]] = 0
    end

    return r
end

function solve(col)
    # indices
    ωxmap = 1:col.np
    ωymap = (col.np+1):2*col.np
    χxmap = (2*col.np+1):3*col.np
    χymap = (3*col.np+1):4*col.np
    N = 4*col.np

    # get LHSs
    A = get_LHS(col)

    # get RHSs
    r = get_RHS(col)

    # solve each column
    sol = A\r
    ωx = FEField(sol[ωxmap], col, col)
    ωy = FEField(sol[ωymap], col, col)
    χx = FEField(sol[χxmap], col, col)
    χy = FEField(sol[χymap], col, col)

    # # exact solution
    # x = col.p[:, 1]
    # z = col.p[:, 2]
    # # u_a = @. -(bx(x, z)*exp(-z)*(-1 + exp(z))*(-1 + exp(H(x) + z)))/(1 + exp(H(x)))
    # u_a = @. exp(-z)*(-1 + exp(z))*(bx(x, z) - (bx(x, z)*exp(H(x))*(-1 + exp(H(x)) - H(x)))/(-1 + exp(H(x)))^2 - (bx(x, z)*exp(H(x) + z)*(-1 + exp(H(x)) - H(x)))/(-1 + exp(H(x)))^2)
    # err = FEField(abs.(u - u_a), col, col)
    # println(@sprintf("Max error: %1.1e", maximum(abs.(u - u_a))))
    # s = ShapeFunctionIntegrals(col.s, col.s)
    # J = Jacobians(col)
    # println(@sprintf("L2 error: %1.1e", L2norm(err, s, J)))

    # # save as .vtu
    # cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, col.t[i, :]) for i ∈ axes(col.t, 1)]
    # vtk_grid("output/pg_vort_DG.vtu", col.p', cells) do vtk
    #     vtk["ωx"] = ωx.values
    #     vtk["ωy"] = ωy.values
    #     vtk["χx"] = χx.values
    #     vtk["χy"] = χy.values
    # end

    z = -H:H/100:0
    ωx_f(z) = evaluate(ωx, [0, z])
    ωy_f(z) = evaluate(ωy, [0, z])
    χx_f(z) = evaluate(χx, [0, z])
    χy_f(z) = evaluate(χy, [0, z])
    fig, ax = subplots(1, 2, figsize=(2*2, 3.2), sharey=true)
    ax[1].plot(ωx_f.(z), z, label=L"\omega^x")
    ax[1].plot(ωy_f.(z), z, label=L"\omega^y")
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


"""
    ie, edge = function vert_edge(p)

Given a triangle with points `p` of the form 
    <| or |>, 
find the local edge index and edge nodes of the vertical edge.
"""
function vert_edge(p)
    for ie=1:3
        edge = [ie, mod1(ie+1, 3)]
        if p[edge[1], 1] == p[edge[2], 1]
            return ie, edge
        end
    end
end

"""
Given a triangle with points `p` and a vertical edge at local edge index `ie`,
find which side the vertical edge is on. Return -1 for left, +1 for right.
"""
function side_of_vert_edge(p, ie)
    if p[mod1(ie+2, 3), 1] > p[ie, 1]
        return -1
    else
        return +1
    end
end

nz = 80
h = 2/(2nz - 3)
println("h = ", h)
p = zeros(2*nz, 2)
p[1, :] = [0 0]
p[2, :] = [h 0]
for i=2:nz-1
    p[2i-1, :] = [0  -(2i-3)*h/2]
    p[2i,   :] = [h  -(i-1)*h]
end
p[2nz-1, :] = [0 -(2nz-3)*h/2]
p[2nz,   :] = [h -(2nz-3)*h/2]

t = [i + j for i=1:2nz-2, j=0:2]
e = [1, 2, 2nz-1, 2nz]

fig, ax = subplots(1, figsize=(1, 3))
tplot(p, t, fig=fig, ax=ax)
ax.axis("equal")
ax.set_ylim(-1.1, 0.1)
savefig("scratch/images/mesh.png")
println("scratch/images/mesh.png")
plt.close()

col = FEGrid(p, t, e, 1)

H = 1
ε² = 0.01
Ux = 0
b(x, z) = x
bx(x, z) = 1
solve(col)