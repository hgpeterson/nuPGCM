using DistMesh, WriteVTK, HDF5, PyPlot, Printf

pygui(false)
plt.style.use("plots.mplstyle")
plt.close("all")

function dparabola(p; its=100, debug=false)
    if p[1] == p[2] == 0 return min(1 + p[2], 1/2*sqrt(3 + 4*p[2]))*sign(p[1]^2 + p[2]^2 - 1 - p[3]) end
    θ = atan(p[2], p[1])
    f(r) = (r*cos(θ) - p[1])^2 + (r*sin(θ) - p[2])^2 + (r^2 - 1 - p[3])^2
    f′(r) = 4r^3 - (4p[3] + 2)r - 2p[1]*cos(θ) - 2p[2]*sin(θ)
    f′′(r) = 12r^2 - 4p[3] - 2
    r = √(p[1]^2 + p[2]^2) + 0.5
    for i ∈ 1:its
        rnew = r - f′(r)/f′′(r)
        if abs(rnew - r) < 1e-16 break end
        if rnew <= 0 
            print(".")
            r += 0.1
        else
            r = rnew
        end
        if debug println("$i $r") end
    end
    if debug
        fig, ax = plt.subplots(1)
        rs = range(0, 2, length=100)
        ax.axhline(0, lw=0.5, c="k", ls="-")
        ax.plot(rs, [f(rs[i]) for i in eachindex(rs)])
        ax.plot(rs, [f′(rs[i]) for i in eachindex(rs)])
        ax.axvline(r, lw=1, c="r", ls="-")
        ax.set_xlabel(L"r")
        ax.set_ylabel(L"f(r)")
        ax.spines["bottom"].set_visible(false)
        savefig("f.png")
        println("f.png")
        plt.close()
    end
    return sqrt(f(r))*sign(r^2 - 1 - p[3])
end

# r = dparabola([0.4, 0, 0], debug=true)

# x = range(-1, 1, length=1000)
# y = zeros(length(x))
# z = zeros(length(x))
# f = [dparabola([x[i], y[i], z[i]]) for i in eachindex(x)]
# fig, ax = plt.subplots(1)
# ax.axhline(0, lw=0.5, c="k", ls="-")
# ax.plot(x, f)
# ax.set_xlabel(L"x")
# ax.set_ylabel(L"d_p(x, 0, 0)")
# ax.spines["bottom"].set_visible(false)
# savefig("dparabola.png")
# println("dparabola.png")
# plt.close()

# signed distance function
dz0(p) = p[3]
# d(p) = max(dparabola(p), dz0(p))
dsphere(p) = sqrt(p[1]^2 + p[2]^2 + (p[3] - 1)^2) - √2
d(p) = max(dsphere(p), dz0(p))

# generate mesh
h = 0.02
# fname = @sprintf("bowl3D_%0.2fdm", h)
fname = @sprintf("bowl3D_%0.2fdm_thin", h)
@time "meshing" dm = distmesh(d, HUniform(), h)

# convert to p, t data structures
p = [dm.points[i][j] for i in 1:size(dm.points, 1), j in 1:3]
t = [dm.tetrahedra[i][j] for i in 1:size(dm.tetrahedra, 1), j in 1:4]

# hmin, hmax
using LinearAlgebra, Statistics
h1 = [norm(p[t[i, 1], :] - p[t[i, 2], :]) for i ∈ axes(t, 1)]
h2 = [norm(p[t[i, 2], :] - p[t[i, 3], :]) for i ∈ axes(t, 1)]
h3 = [norm(p[t[i, 3], :] - p[t[i, 4], :]) for i ∈ axes(t, 1)]
h4 = [norm(p[t[i, 4], :] - p[t[i, 1], :]) for i ∈ axes(t, 1)]
hs = [h1; h2; h3; h4]
@printf("%0.3f ≤ h ≤ %0.3f\n", minimum(hs), maximum(hs))
@printf("mean(h)   = %0.3f\n", mean(hs))
@printf("median(h) = %0.3f\n", median(hs))

# write to VTK
cells = [MeshCell(VTKCellTypes.VTK_TETRA, t[i, :]) for i in axes(t, 1)]
points = p'
vtk_grid(fname*".vtu", points, cells) do vtk
end
println(fname*".vtu")

# save to HDF5
h5open(fname*".h5", "w") do file
    write(file, "p", p)
    write(file, "t", t)
end
println(fname*".h5")