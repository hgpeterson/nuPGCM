using LinearAlgebra, Statistics, HDF5, Printf
using Gridap, GridapGmsh, NonhydroPG
using Gmsh: gmsh
using PyPlot

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

# # compute radius of circumscribed sphere of tetrahedron
# function r_circ(p)
#     # Extract points A, B, C, D
#     A = p[1, :]
#     B = p[2, :]
#     C = p[3, :]
#     D = p[4, :]

#     # Compute the midpoints of the edges
#     M_AB = (A + B) / 2
#     M_AC = (A + C) / 2
#     M_AD = (A + D) / 2

#     # Compute the normal vectors to the faces
#     N_ABC = cross(B - A, C - A)
#     N_ABD = cross(B - A, D - A)
#     N_ACD = cross(C - A, D - A)

#     # Solve the system of equations to find the circumcenter
#     M = [N_ABC'; N_ABD'; N_ACD']
#     b = [dot(N_ABC, M_AB); dot(N_ABD, M_AC); dot(N_ACD, M_AD)]
#     circumcenter = M \ b

#     return norm(circumcenter - A)
# end

# # compute radius of inscribed sphere of tetrahedron
# function r_insc(p)
#     # Extract points A, B, C, D
#     A = p[1, :]
#     B = p[2, :]
#     C = p[3, :]
#     D = p[4, :]

#     # Compute the volume of the tetrahedron
#     M = [(B - A)'; (C - A)'; (D - A)']
#     volume = abs(det(M)) / 6

#     # Compute the areas of the four triangular faces
#     area_ABC = triangle_area(A, B, C)
#     area_ABD = triangle_area(A, B, D)
#     area_ACD = triangle_area(A, C, D)
#     area_BCD = triangle_area(B, C, D)

#     # Compute the total surface area
#     surface_area = area_ABC + area_ABD + area_ACD + area_BCD

#     # Compute the inradius
#     return (3 * volume) / surface_area
# end

# # Function to compute the area of a triangle given its vertices
# function triangle_area(A, B, C)
#     return norm(cross(B - A, C - A)) / 2
# end

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
    θ = zeros(size(t, 1), 12)
    for k ∈ axes(t, 1)
        p1 = p[t[k, 1], :]
        p2 = p[t[k, 2], :]
        p3 = p[t[k, 3], :]
        p4 = p[t[k, 4], :]
        θ[k, :] = inner_angles(p1, p2, p3, p4)
    end
    return sort(θ[:])
end

function print_stats(θ)
    @printf("  %f ≤ θ ≤ %f\n", minimum(θ), maximum(θ))
    @printf("  mean(θ):   %f\n", mean(θ))
    @printf("  median(θ): %f\n", median(θ))
    @printf("  std(θ):    %f\n", std(θ))
end

# distmesh
h = 0.02
fname = @sprintf("bowl3D_%0.2fdm.h5", h)
p = h5read(fname, "p")
t = h5read(fname, "t")
@time θ_dm = inner_angles(p, t)
println("DistMesh:")
print_stats(θ_dm)

# gmsh
fname = @sprintf("meshes/bowl3D_%0.2f.msh", h)
p, t = get_p_t(fname)
@time θ_gm = inner_angles(p, t)
println("Gmsh:")
print_stats(θ_gm)

# plot
fig, ax = plt.subplots(1)
ax.hist(θ_dm, bins=100, density=true, alpha=0.5, label="DistMesh")
ax.hist(θ_gm, bins=100, density=true, alpha=0.5, label="Gmsh")
ax.legend()
ax.set_xlabel("Inner angle (degrees)")
ax.set_ylabel("Density")
ax.set_xlim(0, 120)
ax.set_xticks(0:30:120)
ax.set_ylim(0, 0.05)
savefig(@sprintf("out/inner_angles_%.2f.png", h))
println(@sprintf("out/inner_angles_%.2f.png", h))
plt.close()