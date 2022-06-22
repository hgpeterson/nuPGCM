###
# evolve ζₜ + J(ψ, ζ) = F on FEM grid
###

using nuPGCM, LinearAlgebra, SparseArrays, Printf

Random.seed!(42)

"""
    K = get_K()

Inversion: Δψ = ζ → ψ = K⁻¹ζ
"""
function get_K()
    K = Tuple{Int64,Int64,Float64}[]  
	for k=1:nt
		# calculate contribution to K from element k
        Kᵏ = zeros(n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η, dξ=1) + 
                             shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η, dη=1) 
                Kᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

		# add to global system
		for i=1:n
			for j=1:n
                if t[k, i] in e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(K, (t[k, i], t[k, j], Kᵏ[i, j]))
			end
		end
	end

    # dirichlet ψ = 0 along edges
    for i=1:ne
        push!(K, (e[i], e[i], 1))
    end

    # make CSC matrix
    K = sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), np, np)

    return lu(K)
end

"""
    ψ = invert(ζ)

Compute ψ = K⁻¹ζ
"""
function invert(ζ)
    return K\ζ
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
            func(x, y) = (∂x(ψ, x, y, k)*∂y(ζ, x, y, k) - ∂y(ψ, x, y, k)*∂x(ζ, x, y, k))*shape_func(C₀[k, i, :], x, y)
            J[t[k, i]] += tri_quad(func, p[t[k, 1:3], :]; degree=2)
        end
    end 
    return J
end

function plots(ψ, ζ, i, time)
    plot_horizontal(p, t, ψ; clabel=L"Streamfunction $\psi$", vext=0.8)
    title(latexstring(L"$t = $", @sprintf("%.3f", time)))
    savefig(@sprintf("images/psi%03d.png", i))
    println(@sprintf("images/psi%03d.png", i))
    plt.close()

    plot_horizontal(p, t, ζ; clabel=L"Vorticity $\zeta$", vext=0.5)
    title(latexstring(L"$t = $", @sprintf("%.3f", time)))
    savefig(@sprintf("images/zeta%03d.png", i))
    println(@sprintf("images/zeta%03d.png", i))
    plt.close()

    return i + 1
end

# mesh
p, t, e = load_mesh("../meshes/circle2.h5")

# quad mesh
# p, t, e = add_midpoints(p, t)

# indices
np = size(p, 1)
nt = size(t, 1)
ne = size(e, 1)

# number of nodes per triangle
n = size(t, 2)

# coords 
x = p[:, 1]
y = p[:, 2]

# shape functions
C₀ = get_shape_func_coeffs(p, t)

# K matrix
K = get_K()

# Mass matrix
M = lu(nuPGCM.get_M(p, t, C₀))

function evolve()
    # initial condition
    i_img = 0
    time = 0
    ζ = 0.1*randn(np)
    ψ = invert(ζ)
    i_img = plots(ψ, ζ, i_img, time)

    # timestep
    Δt = 1e-5

    # step forward
    for i=1:10000
        # get ψ
        ψ = invert(ζ)

        # get advection term
        J = get_J(ψ, ζ)

        # euler step
        ζ = ζ - Δt*(M\J)

        # time
        time += Δt

        if i % 100 == 0
            i_img = plots(ψ, ζ, i_img, time)
        end
    end
end

evolve()