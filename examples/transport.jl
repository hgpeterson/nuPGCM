using SparseArrays, PyPlot

nx = 1025
ny = 513

τ0 = .1
ρ0 = 1e3
f0 = 1e-4
β = 2e-11
r = 5e-6
H0 = 4000.
Lx = 10e6
Ly = 5e6

Δx = Lx/(nx-1)
Δy = Ly/(ny-1)

x = (0:(nx-1))*Δx
y = (0:(ny-1))*Δy

f(y) = f0 + β*y

# flat bottom
H(x, y) = H0
Hx(x, y) = 0
Hy(x, y) = 0

# sloping sides
#d = 500e3
#H(x, y) = H0*(1 - exp(-x/d) - exp((x-Lx)/d) + 2exp(-Lx/d))
#Hx(x, y) = H0*(exp(-x/d) - exp((x-Lx)/d))/d
#Hy(x, y) = 0

τ(y) = -τ0*cos(π*y/Ly)
τy(y) = π*τ0/Ly*sin(π*y/Ly)

Qx(x, y) = -f(y)*Hx(x, y)/H(x, y)^2
Qy(x, y) = β/H(x, y) - f(y)*Hy(x, y)/H(x, y)^2

function insert!(i, j, k, l, v, I, J, V)
  append!(I, [(i-1)*ny+j])
  append!(J, [(k-1)*ny+l])
  append!(V, [v])
end

I = Array{Float64}(undef, 0)
J = Array{Float64}(undef, 0)
V = Array{Float64}(undef, 0)
b = Array{Float64}(undef, 0)
for i = 1:nx, j = 1:ny
  if (i == 1) | (i == nx) | (j == 1) | (j == ny)
    insert!(i, j, i, j, 1, I, J, V)
    append!(b, [0])
  else
    # advection
    insert!(i, j, i, j+1, -Qx(x[i], y[j])/2Δy, I, J, V)
    insert!(i, j, i, j-1, Qx(x[i], y[j])/2Δy, I, J, V)
    insert!(i, j, i+1, j, Qy(x[i], y[j])/2Δx, I, J, V)
    insert!(i, j, i-1, j, -Qy(x[i], y[j])/2Δx, I, J, V)
    # friction
    insert!(i, j, i, j, -2r*(1/Δx^2+1/Δy^2)/H(x[i], y[j]), I, J, V)
    insert!(i, j, i+1, j, r/Δx^2/H(x[i], y[j]), I, J, V)
    insert!(i, j, i-1, j, r/Δx^2/H(x[i], y[j]), I, J, V)
    insert!(i, j, i, j+1, r/Δy^2/H(x[i], y[j]), I, J, V)
    insert!(i, j, i, j-1, r/Δy^2/H(x[i], y[j]), I, J, V)
    insert!(i, j, i+1, j, -r*Hx(x[i], y[j])/2Δx/H(x[i], y[j])^2, I, J, V)
    insert!(i, j, i-1, j, r*Hx(x[i], y[j])/2Δx/H(x[i], y[j])^2, I, J, V)
    insert!(i, j, i, j+1, -r*Hy(x[i], y[j])/2Δy/H(x[i], y[j])^2, I, J, V)
    insert!(i, j, i, j-1, r*Hy(x[i], y[j])/2Δy/H(x[i], y[j])^2, I, J, V)
    # forcing
    append!(b, [-τy(y[j])/ρ0/H(x[i], y[j]) + Hy(x[i], y[j])*τ(y[j])/ρ0/H(x[i], y[j])^2])
  end
end
A = sparse(I, J, V)

ψ = reshape(A\b, (ny, nx))

fig, ax = subplots(3, 1, sharex=true, figsize=(6.4, 7.8), gridspec_kw=Dict("height_ratios"=>(1, 4, 4)))
ax[1].plot(1e-3x, -H.(x, 0))
ax[2].contour(1e-3x, 1e-3y, f.(y)./H.(x', y), levels=f0/H0 .+ β*Ly/H0*(.05:.1:.95))
ax[3].contour(1e-3x, 1e-3y, ψ, levels=(.06:.06:.6)*π*τ0/ρ0/β*Lx/Ly)
ax[2].set_aspect(1)
ax[3].set_aspect(1)
ax[1].set_ylim(-1.1H0, 0)
ax[1].set_ylabel("height (m)")
ax[2].set_ylabel("meridional coordinate (km)")
ax[3].set_ylabel("meridional coordinate (km)")
ax[3].set_xlabel("zonal coordinate (km)")
ax[1].set_title("depth profile")
ax[2].set_title("\$f/H\$ contours")
ax[3].set_title("streamfunction")
ax[1].set_yticks([-H0, 0])
fig.tight_layout()
fig.align_ylabels()
savefig("debug.png")
