###
# Evolve 
#
#   ∂t(ζ) + J(ψ, ζ) + β ∂y(ψ) = ν ∇²ζ 
#                  ∇²ψ + ψ/λ² = ζ
#
# on FEM grid.
###

using nuPGCM, PyPlot, Random, LinearAlgebra, SparseArrays, Printf, ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

Random.seed!(42)

"""
    K = get_K()

Diffusion term: Kᵢⱼ = -∫ [∂x(φᵢ)∂x(φⱼ) + ∂y(φᵢ)∂y(φⱼ)]
"""
function get_K()
    K = Tuple{Int64,Int64,Float64}[]  
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

		# add to global system
		for i=1:n
			for j=1:n
                push!(K, (t[k, i], t[k, j], Kᵏ[i, j]))
			end
		end
	end

    # make CSC matrix
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), np, np)

    return K
end

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
    Js = get_Js()

Compute Jᵢⱼₖ = ∫ [∂x(φₖ)∂y(φⱼ) - ∂y(φₖ)∂x(φⱼ)] φᵢ. 
"""
function get_Js()
    Js = zeros(nt, n, n, n)
    @showprogress "Computing Js..." for k₀=1:nt
        for i=1:n
            for j=1:n
                for k=1:n
                    func(x, y) = (shape_func(C₀[k₀, k, :], x, y; dξ=1)*shape_func(C₀[k₀, j, :], x, y; dη=1) -
                                  shape_func(C₀[k₀, k, :], x, y; dη=1)*shape_func(C₀[k₀, j, :], x, y; dξ=1))*
                                  shape_func(C₀[k₀, i, :], x, y)
                    Js[k₀, i, j, k] = tri_quad(func, p[t[k₀, 1:3], :]; degree=4)
                end
            end
        end
    end
    return Js
end

"""
    J = get_J(ψ, ζ)

Compute Jᵢ = Jᵢⱼₖ ψₖ ζⱼ + β ψₖ ∂x(φₖ)*φᵢ
"""
function get_J(ψ, ζ)
    J = zeros(np)
    for k=1:nt
        J[t[k, :]] += reshape(reshape(Js[k, :, :, :], n^2, n)*ψ[t[k, :]], n, n)*ζ[t[k, :]]
    end
    J += β*Cx*ψ
    return J
end

function plots(ψ, ζ, i, time)
    fig, ax, im = plot_horizontal(p, t, ψ; clabel=L"Streamfunction $\psi$")
    ax.set_title(latexstring(L"$t = $", @sprintf("%df", time)))
    ax.set_yticks(-1:0.5:1)
    savefig(@sprintf("images/psi%03d.png", i), dpi=200)
    # println(@sprintf("images/psi%03d.png", i))
    plt.close()

    fig, ax, im = plot_horizontal(p, t, ζ; clabel=L"Vorticity $\zeta$", vext=1, contours=false)
    ax.set_title(latexstring(L"$t = $", @sprintf("%d", time)))
    ax.set_yticks(-1:0.5:1)
    savefig(@sprintf("images/zeta%03d.png", i), dpi=200)
    # println(@sprintf("images/zeta%03d.png", i))
    plt.close()

    return i + 1
end

"""
    y, t = advect(f, y, t, Δt)

Apply RK4 timestep with dt(y) = f(t, y).
"""
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

"""
    f_adv(t, ζ)

RHS of ζ equation.
"""
function f_adv(t, ζ)
    ψ = invert(ζ) 
    J = get_J(ψ, ζ)
    return -(M_LU\J)
end

# mesh
# p, t, e = load_mesh("../meshes/circle1.h5")
p, t, e = load_mesh("../meshes/circle2.h5")

# quad mesh
# p, t, e = add_midpoints(p, t)

# indices
np = size(p, 1)
nt = size(t, 1)
ne = size(e, 1)

# beta
# β = 1
β = 0

# Rossby radius
# λ = 0.5
λ = Inf

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

# derivative matrices
Cx, Cy = nuPGCM.get_Cξ_Cη(p, t, C₀)

# diffusion matrix 
K = get_K()

# diffusion coefficient
ν = 1e-5

# J matrices
Js = get_Js()

function evolve()
    # initial condition
    time = 0
    # ζ = 0.4*randn(np)
    Δ = 0.1
    # ζ = @. (exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) - exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)))
    ζ = @. (exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) + exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)))
    ψ = invert(ζ)
    i_img = plots(ψ, ζ, 0, time)

    # timestep
    Δt = 1e-1

    # step forward
    @showprogress "Integrating in time..." for i=1:500
        ζ, time = advect(f_adv, ζ, time, Δt)

        if i % 10 == 0
            ψ = invert(ζ)
            i_img = plots(ψ, ζ, i_img, time)
        end
    end
end

evolve()