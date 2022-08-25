using nuPGCM
using PyPlot
using PyCall

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

function delaunay(p)
    tri = pyimport("matplotlib.tri")
    t = tri[:Triangulation](p[:,1], p[:,2])
    return Int64.(t[:triangles] .+ 1)
end

L = 5e6
nξ = 2^4
ξ = -L:2L/(nξ - 1):L
Δ = L/5
G(x) = 1 - exp(-x^2/(2*Δ^2)) 
Gx(x) = x/Δ^2*exp(-x^2/(2*Δ^2))
H₀ = 2e3
H  = @. H₀*G(ξ - L)*G(ξ + L)
Hx = @. H₀*(Gx(ξ - L)*G(ξ + L) + G(ξ - L)*Gx(ξ + L))

dz = 200
nz = ones(Int64, nξ)
for i=2:nξ-1
    nz[i] += Int64(ceil(H[i]/dz))
end

z = zeros(Int64(sum(nz)))
p = zeros(size(z, 1), 2)
c = 1
fig, ax = subplots()
ax.plot(ξ[1],   0, ".", c="tab:blue")
ax.plot(ξ[end], 0, ".", c="tab:blue")
p[1, :] = [ξ[1] 0]
p[end, :] = [ξ[end] 0]
for i=2:nξ-1
    n = nz[i]
    for j=1:n
        z[c + j] = -H[i]*(1 - (j - 1)/(n - 1))
        p[c + j, :] = [ξ[i] z[c + j]]
        ax.plot(ξ[i], z[c + j], ".", c="tab:blue")
    end
    global c += n
end
savefig("images/debug.png")
plt.close()

t = delaunay(p)
tplot(p, t)
savefig("images/debug.png")
plt.close()