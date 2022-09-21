using nuPGCM
using PyPlot
using SparseArrays
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
    u, p = solve_stokes()

Stokes problem:
    -Δu + ∇p = 0 on Ω,
         ∇⋅u = 0 on Ω,
           u = 0 on Γ,
with extra condition
    ∫ p dx = 0.
Here u = (u₁, u₂) is the velocity vector and p is the pressure.
Weak form:
    ∫ (∇u)⋅(∇v) - p (∇⋅v) dx = 0
    ∫ q (∇⋅u) dx = 0
Or just,
    ∫ (∇u)⋅(∇v) - p (∇⋅v) + q (∇⋅u) dx = 0
for all 
    v ∈ V = {(v₁, v₂) | vᵢ ∈ P₂}
    q ∈ Q = {q ∈ P₁ | ∫ q dx = 0}
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes(g₁::Grid, g₂::Grid, sfi_uu::ShapeFunctionIntegrals, sfi_up::ShapeFunctionIntegrals, sfi_pu::ShapeFunctionIntegrals, J::Jacobians)
    A = Tuple{Int64,Int64,Float64}[]
    N = 2*g₂.np + g₁.np
    b = zeros(N)
    for k=1:g₁.nt
        # contribution from (∇u)⋅(∇v) term
        Kᵏ = abs(J.J[k])*(sfi_uu.φξφξ.*(J.ξx[k]^2       + J.ξy[k]^2) + 
                          sfi_uu.φξφη.*(J.ξx[k]*J.ηx[k] + J.ξy[k]*J.ηy[k]) +
                          sfi_uu.φηφξ.*(J.ηx[k]*J.ξx[k] + J.ηy[k]*J.ξy[k]) +
                          sfi_uu.φηφη.*(J.ηx[k]^2       + J.ηy[k]^2))

        # contribution from -p (∇⋅v) term

        # contribution from q (∇⋅u) term

        # add to global system
        for i=1:g.nn
            if g.t[k, i] in g.e
                # edge node, leave for dirichlet
                continue
            end
            for j=1:g.nn
                push!(A, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
            end
        end
    end
    # dirichlet along edges
    for i in g.e
        push!(K, (i, i, 1))
    end
    b[g.e] = u₀

    # make CSC matrix
    K = sparse((x -> x[1]).(K), (x -> x[2]).(K), (x -> x[3]).(K), g.np, g.np)

    # solve
    return K\b
end

"""
    u, p = stokes_res(nref, order)
"""
function stokes_res(nref; plot=false)
    # geometry type
    # geo = "square"
    geo = "circle"

    # get shape functions
    sf_u = ShapeFunctions(2; zeromean=false)
    sf_p = ShapeFunctions(1; zeromean=true)

    # get shape function integrals
    sfi_uu = ShapeFunctionIntegrals(sf_u, sf_u)
    sfi_up = ShapeFunctionIntegrals(sf_u, sf_p)
    sfi_pu = ShapeFunctionIntegrals(sf_p, sf_u)

    # get grids
    g₁ = Grid("../meshes/$geo/mesh$nref.h5", 1)
    g₂ = Grid("../meshes/$geo/mesh$nref.h5", 2)

    # get Jacobians
    J = Jacobians(g₁)

    # # mesh resolution 
    # h = 1/sqrt(g₂.np)

    # solve stokes problem
    u, p = solve_stokes(g₁, g₂, sfi_uu, sfi_up, sfi_pu, J)

    if plot
        fig, ax, im = tplot(g₂.p, g₂.t, u[1, :])
        cb = colorbar(im, ax=ax, label=L"u_1")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/u1.png")
        println("images/u1.png")
        plt.close()
        fig, ax, im = tplot(g₂.p, g₂.t, u[2, :])
        cb = colorbar(im, ax=ax, label=L"u_2")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/u2.png")
        println("images/u2.png")
        plt.close()
        fig, ax, im = tplot(g₁.p, g₁.t, p)
        cb = colorbar(im, ax=ax, label=L"p")
        ax.axis("equal")
        ax.set_xlabel(L"x")
        ax.set_ylabel(L"y")
        savefig("images/p.png")
        println("images/p.png")
        plt.close()
    end

    return u, p
end