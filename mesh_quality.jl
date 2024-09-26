using LinearAlgebra, Statistics, HDF5, Printf
using Gridap, GridapGmsh, NonhydroPG
using Gmsh: gmsh
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

"""
    θ = angle(v1, v2)

Compute angle (in degrees) between two vectors `v1` and `v2`.
"""
function angle(v1, v2)
    return 180/π*acos(dot(v1, v2) / (norm(v1) * norm(v2)))
end

"""
    θ = angle(p1, p2, p3)

Compute angle (in degrees) between two vectors defined by v1 = `p1` - `p2` and 
v2 = `p3` - `p2`.
"""
function angle(p1, p2, p3)
    v1 = p1 - p2
    v2 = p3 - p2
    return angle(v1, v2)
end

"""
    [θ1, θ2, θ3] = inner_angles(p1, p2, p3)

Compute the inner angles of a triangle defined by the vertices `p1`, `p2`, 
and `p3`.
"""
function inner_angles(p1, p2, p3)
    θ1 = angle(p2, p1, p3)
    θ2 = angle(p1, p2, p3)
    θ3 = angle(p2, p3, p1)
    return [θ1, θ2, θ3]
end

"""
    [θ1, θ2, θ3, θ4, θ5, θ6, θ7, θ8, θ9, θ10, θ11, θ12] = inner_angles(p1, p2, p3, p4)

Compute the inner angles of a tetrahedron defined by the vertices `p1`, `p2`,
`p3`, and `p4`.
"""
function inner_angles(p1, p2, p3, p4)
    θ1, θ2, θ3 = inner_angles(p1, p2, p3)
    θ4, θ5, θ6 = inner_angles(p1, p2, p4)
    θ7, θ8, θ9 = inner_angles(p1, p3, p4)
    θ10, θ11, θ12 = inner_angles(p2, p3, p4)
    return [θ1, θ2, θ3, θ4, θ5, θ6, θ7, θ8, θ9, θ10, θ11, θ12]
end

"""
    θ = inner_angles(p, t)

Compute (and sort) the inner angles of a tetrahedral mesh defined by the 
vertices `p` and connectivities `t`.
"""
function inner_angles(p, t)
    n = size(t, 2) # number of points per element (3 for tri, 4 for tet)
    θ = zeros(size(t, 1), n == 3 ? 3 : 12) # 3 inner angles per tri, 12 per tet
    for k ∈ axes(t, 1)
        pts = (p[t[k, i], :] for i ∈ 1:n)
        θ[k, :] = inner_angles(pts...)
    end
    return sort(θ[:])
end

function print_stats(θ)
    @printf("  %f ≤ θ ≤ %f\n", minimum(θ), maximum(θ))
    @printf("  mean(θ):   %f\n", mean(θ))
    @printf("  median(θ): %f\n", median(θ))
    @printf("  std(θ):    %f\n", std(θ))
end

# # distmesh
# h = 0.01
# fname = @sprintf("bowl3D_%0.2fdm.h5", h)
# p = h5read(fname, "p")
# t = h5read(fname, "t")
# @time θ_dm = inner_angles(p, t)
# println("DistMesh:")
# print_stats(θ_dm)

# gmsh
# fname = @sprintf("meshes/bowl2D_%0.2f.msh", h)
# fname = "meshes/bowl2D_exp.msh"
fname = "meshes/bowl2D_0.01.msh"
p, t = get_p_t(fname)
@time θ_gm = inner_angles(p, t)
println("Gmsh:")
print_stats(θ_gm)

# plot
fig, ax = plt.subplots(1)
# ax.hist(θ_dm, bins=100, density=true, alpha=0.5, label="DistMesh")
ax.hist(θ_gm, bins=100, density=true, alpha=0.5, label="Gmsh")
# ax.legend()
ax.set_xlabel("Inner angle (degrees)")
ax.set_ylabel("Density")
ax.set_xlim(0, 120)
ax.set_xticks(0:30:120)
ax.set_ylim(0, 0.05)
# ax.set_yscale("log")
# ax.set_ylim(1e-5, 1e0)
# savefig(@sprintf("out/inner_angles_%.2f.png", h))
# println(@sprintf("out/inner_angles_%.2f.png", h))
savefig(@sprintf("out/inner_angles_%s.png", fname[8:end-4]))
println(@sprintf("out/inner_angles_%s.png", fname[8:end-4]))
plt.close()