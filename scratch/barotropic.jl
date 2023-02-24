## Solve
##     -∂x(r_sym ∂x(Ψ)) - ∂y(r_sym ∂y(Ψ)) - 
##         ∂x(r_asym ∂y(Ψ)) - ∂y(r_asym ∂x(Ψ)) +
##             ∂y(f/H)∂x(Ψ) - ∂x(f/H)∂y(Ψ) 
##     =
##     -J(1/H, γ) + ε² z⋅(∇×τ/H) - ε² ∇⋅(ν*(ωb + τʲω_τⱼ)/H)
## with Ψ = 0 on boundary.

using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

function solve_barotropic(g, r_sym, r_asym)
    # indices
    N = g.np

    # unpack
    bdy = g.e["bdy"]
    J = g.J
    s = g.sfi

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    rhs = zeros(N)
    for k=1:g.nt
        # matrices
        JJ = J.Js[k, :, :]*J.Js[k, :, :]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        K′ = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]

        # interior terms
        for i=1:g.nn, j=1:g.nn
            if g.t[k, i] ∉ bdy 
                push!(A, (g.t[k, i], g.t[k, j], r_sym*K[i, j]))
                push!(A, (g.t[k, i], g.t[k, j], r_asym*K′[i, j]))
            end
        end

        # rhs
        rhs[g.t[k, :]] += J.dets[k]*s.M*ones(g.nn)
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
        rhs[i] = 0
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    # solve
    return FEField(A\rhs, g)
end

function main(; order)
    g = FEGrid("meshes/circle/mesh3.h5", order)
    r_sym = -0.1
    r_asym = +3.0
    Ψ = solve_barotropic(g, r_sym, r_asym)

    fig, ax, im = tplot(Ψ)
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"y")
    ax.axis("equal")
    colorbar(im, ax=ax, label=L"\Psi")
    savefig("scratch/images/psi.png")
    println("scratch/images/psi.png")
    plt.close()
end

main(order=2)