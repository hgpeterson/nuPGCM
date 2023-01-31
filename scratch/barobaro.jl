using nuPGCM
using PyPlot

include("utils.jl")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

# params
ε² = 1
β = 0

# grid
g = FEGrid("../meshes/circle/mesh1.h5", 2)
g1 = FEGrid("../meshes/circle/mesh1.h5", 1)

# depth
H = FEField(ones(g.np), g, g1)
Hx = FEField(zeros(g.np), g, g1)
Hy = FEField(zeros(g.np), g, g1)

# bottom drag
r_sym = FEField(ones(g.np), g, g1)
r_asym = FEField(zeros(g.np), g, g1)

# LHS
LHS = get_barotropic_LHS(g1, g, ε², β, H, Hx, Hy, r_sym, r_asym)

# RHS
RHS = ones(g.np)
RHS[g.e] .= 0
Ψ = FEField(LHS\RHS, g, g1)

# plot
fig, ax, im = tplot(g.p, g.t, Ψ.values/1e6)
ax.set_xlabel(L"x")
ax.set_ylabel(L"y")
ax.axis("equal")
colorbar(im, ax=ax, label=L"\Psi")
savefig("images/psi.png")
println("images/psi.png")
plt.close()