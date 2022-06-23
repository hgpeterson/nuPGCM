###
# evolve ζₜ + J(ψ, ζ) = F on FEM grid
###

using nuPGCM, PyPlot, Random, LinearAlgebra, SparseArrays, Printf, ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

Random.seed!(42)

"""
    A = get_A()

Inversion: ψ = A⁻¹ζ
"""
function get_A()
    A = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to K from element k
        Kᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = -shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η, dξ=1) - 
                              shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η, dη=1) 
                Kᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        if λ < Inf
		    # calculate contribution to L from element k
            Lᵏ = zeros(n, n)
            for i=1:n
                for j=1:n
                    func(ξ, η) = -1/λ^2*shape_func(C₀[k, j, :], ξ, η)*shape_func(C₀[k, i, :], ξ, η) 
                    Lᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
                end
            end
        end

		# add to global system
		for i=1:n
			for j=1:n
                if t[k, i] in e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(A, (t[k, i], t[k, j], Kᵏ[i, j]))
                if λ < Inf
                    push!(A, (t[k, i], t[k, j], Lᵏ[i, j]))
                end
			end
		end
	end

    # dirichlet ψ = 0 along edges
    for i=1:ne
        push!(A, (e[i], e[i], 1))
    end

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), np, np)

    return lu(A)
end

"""
    ψ = invert(ζ)

Compute ψ = A⁻¹ζ
"""
function invert(ζ)
    rhs = M*ζ
    rhs[e] .= 0 # ψ = 0 on bdy
    return A\rhs
end

"""
Wrapper functions for derivatives.
"""
function ∂x(v, x, y, k)
    return ∂ξ(v, x, y, k, p, t, C₀)
end
function ∂y(v, x, y, k)
    return ∂η(v, x, y, k, p, t, C₀)
end

"""
    J = get_J(ψ, ζ)

Compute Jᵢ = ∫ [∂x(ψ)∂y(ζ) - ∂y(ψ)∂x(ζ)] φᵢ. 
"""
function get_J(ψ, ζ)
    J = zeros(np) 
    for k=1:nt
        for i=1:n
            func(x, y) = (∂x(ψ, x, y, k)*∂y(ζ, x, y, k) - ∂y(ψ, x, y, k)*∂x(ζ, x, y, k) + β*∂x(ψ, x, y, k))*shape_func(C₀[k, i, :], x, y)
            J[t[k, i]] += tri_quad(func, p[t[k, 1:3], :]; degree=4)
        end
    end 
    return J
end

function plots(ψ, ζ, i, time)
    fig, ax, im = plot_horizontal(p, t, ψ; clabel=L"Streamfunction $\psi$")
    ax.set_title(latexstring(L"$t = $", @sprintf("%.3f", time)))
    ax.set_yticks(-1:0.5:1)
    savefig(@sprintf("images/psi%03d.png", i), dpi=200)
    # println(@sprintf("images/psi%03d.png", i))
    plt.close()

    fig, ax, im = plot_horizontal(p, t, ζ; clabel=L"Vorticity $\zeta$")
    ax.set_title(latexstring(L"$t = $", @sprintf("%.3f", time)))
    ax.set_yticks(-1:0.5:1)
    savefig(@sprintf("images/zeta%03d.png", i), dpi=200)
    # println(@sprintf("images/zeta%03d.png", i))
    plt.close()

    return i + 1
end

function advect(f, y, t, Δt)
    # RK4
    k₁ = f(t,        y)
    k₂ = f(t + Δt/2, y + Δt*k₁/2)
    k₃ = f(t + Δt/2, y + Δt*k₂/2)
    k₄ = f(t + Δt,   y + Δt*k₃)
    y += Δt/6*(k₁ + 2*k₂ + 2*k₃ + k₄)
    t += Δt
    return y, t
end

function f_adv(t, ζ)
    ψ = invert(ζ) 
    J = get_J(ψ, ζ)
    return -(M_LU\J)
end

# mesh
p, t, e = load_mesh("../meshes/circle1.h5")
# p, t, e = load_mesh("../meshes/circle2.h5")

# quad mesh
# p, t, e = add_midpoints(p, t)

# indices
np = size(p, 1)
nt = size(t, 1)
ne = size(e, 1)

# beta
β = 1

# Rossby radius
λ = 0.5
# λ = Inf

# number of nodes per triangle
n = size(t, 2)

# coords 
x = p[:, 1]
y = p[:, 2]

# shape functions
C₀ = get_shape_func_coeffs(p, t)

# inversion matrix
A = get_A()

# mass matrix
M = nuPGCM.get_M(p, t, C₀)
M_LU = lu(M)

function evolve()
    # initial condition
    time = 0
    ζ = 0.5*randn(np)
    # Δ = 0.1
    # ζ = @. 0.9*(exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) - exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)))
    ψ = invert(ζ)
    i_img = plots(ψ, ζ, 0, time)

    # timestep
    Δt = 1e-1

    # step forward
    @showprogress for i=1:1000
        ζ, time = advect(f_adv, ζ, time, Δt)

        if i % 10 == 0
            ψ = invert(ζ)
            i_img = plots(ψ, ζ, i_img, time)
        end
    end
end

evolve()