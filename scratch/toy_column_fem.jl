using nuPGCM
using PyPlot
using PyCall
using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
    t = delaunay(p)

Delaunay triangulation `t` of N x 2 node array `p`.
"""
function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri[:Triangulation](p[:,1], p[:,2])
    return Int64.(t[:triangles] .+ 1)
end

"""
    e = boundary_nodes(t)

Find all boundary nodes `e` in the triangulation `t`.
"""
function boundary_nodes(t)
    edges, boundary_indices = nuPGCM.all_edges(t)
    return unique(edges[boundary_indices,:][:])
end

"""
    inside = inpolygon(p, pv)

Determine if each point in the N x 2 node array `p` is inside the polygon
described by the NE x 2 node array `pv`.
"""
function inpolygon(p::Array{Float64,2}, pv::Array{Float64,2})
    path = pyimport("matplotlib.path")
    poly = path[:Path](pv)
    return [poly[:contains_point](p[ip,:]) for ip = 1:size(p,1)]
end

function edge_midpoints(p, t)
    pmid = reshape(p[t,:] + p[t[:,[2,3,1]],:], :, 2) / 2
    return unique(pmid, dims=1)
end

function remove_tiny_tris(p, t)
    areas = zeros(size(t, 1))
    for i in eachindex(areas)
        areas[i] = tri_area(p[t[i, :], :])
    end
    return t[areas .> 1e-14,:]
end

function remove_outside_tris(p, t, pv)
    pmid = dropdims(sum(p[t,:], dims=2), dims=2) / 3
    is_inside = inpolygon(pmid, pv)
    return t[is_inside,:]
end

function get_grid(L::FT, H₀::FT; res=1, nref=1, vm=false, debug=false) where FT <: Real
    if debug
        p, t, e = load_mesh("../meshes/mesh.h5")
        # p, t, e = load_mesh("../meshes/mesh$nref.h5")
    else
        if vm
            p, t, e = load_mesh("../meshes/bowl_vm$res.h5")
        else
            p, t, e = load_mesh("../meshes/bowl$res.h5")
        end
    end

    # refinements
    
    # get edge nodes (in proper order)
    pv = p[e, :]
    pv = pv[pv[:, 2] .< 0, :]
    pv = sortslices(pv, dims=1)
    pv = [pv; 1 0; -1 0; pv[1, 1] pv[1, 2]]

    for i=1:nref
        # add midpoints
        p = [p; edge_midpoints(p, t)]

        # retriangulate
        t = delaunay(p)
        t = remove_tiny_tris(p, t)
        t = remove_outside_tris(p, t, pv)

        # recompute boundary nodes
        e = boundary_nodes(t)
    end

    # rescale
    p[:, 1] *= L
    p[:, 2] *= H₀

    # shape function coefficients
    C₀ = get_shape_func_coeffs(p, t)

    # t dictionary
    # t_dict = get_t_dict(p, t)
    t_dict = nothing
    return p, t, e, C₀, t_dict
end

"""
    A, b = get_A_b(p, t, e, C₀, δ)

Get matrix A and vector b to solve linear system representing the problem
    -δ² u_zz + u = 1
in a finite element basis.
"""
function get_A_b(p::AbstractMatrix{FT}, t::AbstractMatrix{IT}, e::AbstractVector{IT},
                 C₀::AbstractArray{FT,3}, δ_x::FT, δ_z::FT) where {FT <: Real, IT <: Integer}
    np = size(p, 1)
    nt = size(t, 1)
    n = size(t, 2)
    A = Tuple{IT,IT,FT}[]
    b = zeros(FT, np)
    for k=1:nt        
        # calculate contribution to K from element k
        Kᵏ = zeros(FT, n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = δ_x^2*shape_func(C₀[k, j, :], ξ, η; dξ=1)*shape_func(C₀[k, i, :], ξ, η; dξ=1) +
                             δ_z^2*shape_func(C₀[k, j, :], ξ, η; dη=1)*shape_func(C₀[k, i, :], ξ, η; dη=1)
                Kᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to M from element k
        Mᵏ = zeros(FT, n, n)
        for i=1:n
            for j=1:n
                func(ξ, η) = shape_func(C₀[k, j, :], ξ, η)*shape_func(C₀[k, i, :], ξ, η)
                Mᵏ[i, j] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
            end
        end

        # calculate contribution to b from element k
        bᵏ = zeros(FT, n)
        for i=1:n
            func(ξ, η) = 1*shape_func(C₀[k, i, :], ξ, η)
            bᵏ[i] = tri_quad(func, p[t[k, 1:3], :]; degree=4)
        end

        # add to global system
        for i=1:n
            for j=1:n
                if t[k, i] in e
                    # edge node, leave for dirichlet
                    continue
                end
                push!(A, (t[k, i], t[k, j], Kᵏ[i, j]))
                push!(A, (t[k, i], t[k, j], Mᵏ[i, j]))
            end
            b[t[k, i]] += bᵏ[i]
        end
    end

    # dirichlet u = 0 along edges
    for i in e
        push!(A, (i, i, 1))
    end
    b[e] .= 0

    # make CSC matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), np, np)

    return A, b
end

function solve(p, t, e, C₀, δ_x, δ_z)
    A, b = get_A_b(p, t, e, C₀, δ_x, δ_z)
    return A\b
end

function convergence(nrefs; plots=false)
    # params
    L = 5e6
    H₀ = 2e3
    δ_x = L/10
    # δ_x = 0.
    δ_z = H₀/10

    # highest resolution
    p_fine, t_fine, e_fine, C₀_fine, t_dict_fine = get_grid(L, H₀; debug=true, nref=nrefs[end])
    u_fine = solve(p_fine, t_fine, e_fine, C₀_fine, δ_x, δ_z)

    # np of coarsest resolution
    p_coarse, t_coarse, e_coarse, C₀_coarse, t_dict_coarse = get_grid(L, H₀; debug=true, nref=0)
    np_coarse = size(p_coarse, 1)

    # save errors
    hs = zeros(size(nrefs, 1) - 1)
    errors = zeros(size(nrefs, 1) - 1)
    for k=1:size(nrefs, 1)-1
        nref = nrefs[k]
        println("refinement ", nref)

        # grid
        p, t, e, C₀, t_dict = get_grid(L, H₀; debug=true, nref=nref)
        np = size(p, 1)
        hs[k] = sqrt(np)

        # solve
        u = solve(p, t, e, C₀, δ_x, δ_z)
        if plots
            fig, ax, im = tplot(p/1e3, t, u)
            cb = colorbar(im, ax=ax, label=L"$u$")
            ax.set_xlabel(L"Zonal coordinate $x$ (km)")
            ax.set_ylabel(L"Vertical coordinate $z$ (km)")
            savefig("images/u.png")
            println("images/u.png")
            plt.close()
        end

        # compute error
        # abs_err = zeros(np)
        # @showprogress "Evaluating error..." for i=1:np
        #     abs_err[i] = abs(u[i] - fem_evaluate(u_fine, p[i, 1], p[i, 2], p_fine, t_fine, C₀_fine))
        # end
        abs_err = abs.(u - u_fine[1:np])
        errors[k] = maximum(abs_err)
        if plots
            fig, ax, im = tplot(p/1e3, t, abs_err)
            cb = colorbar(im, ax=ax, label="Absolute error")
            ax.set_xlabel(L"Zonal coordinate $x$ (km)")
            ax.set_ylabel(L"Vertical coordinate $z$ (km)")
            savefig("images/abs_err.png")
            println("images/abs_err.png")
            plt.close()
        end
    end

    if size(nrefs, 1) > 1
        fig, ax = subplots(1)
        ax.set_xlabel(L"$\sqrt{N}$")
        ax.set_ylabel("Error")
        conv = log2(errors[end-1]) - log2(errors[end])
        println(@sprintf("\nConvergence: O(h^%1.1f)\n", conv))
        ax.loglog(hs, errors, "o-", label=@sprintf("order: %1.1f", conv))
        ax.legend()
        savefig("images/conv.png")
        println("images/conv.png")
        plt.close()
    end

    return errors
end

function u_exact(z, δ, H)
    return (exp(-z/δ) - exp(H/δ))*(exp(z/δ) - 1)/(1 + exp(H/δ))
end

# errors = convergence(2; plots=true)
errors = convergence(0:3; plots=true)