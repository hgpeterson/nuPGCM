using nuPGCM
using PyPlot
using PyCall
using LinearAlgebra
using SparseArrays

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

crs = pyimport("cartopy.crs")

if !isdir("../output")
    mkdir("../output")
end
set_out_folder("../output")
if !isdir("$out_folder/data")
    mkdir("$out_folder/data")
end
if !isdir("$out_folder/images")
    mkdir("$out_folder/images")
end

function build_barotropic_LHS(r, H)
    # unpack
    g = r.g
    bdy = g.e["bdy"]
    J = g.J
    el = g.el

    # FEField for f
    Ω = 2π/(24*60*60)
    f = FEField(x -> 2*Ω*sin(ϕ(x)), g)

    # indices
    N = g.np

    # integrands
    function ∫K(ξ, i, j, k)
        ∇φ_i = φξ(el, ξ, i)*J.Js[k, 1, :] + φη(el, ξ, i)*J.Js[k, 2, :]
        ∇φ_j = φξ(el, ξ, j)*J.Js[k, 1, :] + φη(el, ξ, j)*J.Js[k, 2, :]
        return -r(ξ, k)/H(ξ, k)*dot(∇φ_i, ∇φ_j)*J.dets[k]
    end
    function ∫C(ξ, i, j, k)
        # ∇φ_j = φξ(el, ξ, j)*J.Js[k, 1, :] + φη(el, ξ, j)*J.Js[k, 2, :]
        φx_j = φξ(el, ξ, j)*J.Js[k, 1, 1] + φη(el, ξ, j)*J.Js[k, 2, 1]
        φy_j = φξ(el, ξ, j)*J.Js[k, 1, 2] + φη(el, ξ, j)*J.Js[k, 2, 2]
        φ_i = φ(g.el, ξ, i)
        return -((H(ξ, k)*∂(f, ξ, k, 1) - f(ξ, k)*∂(H, ξ, k, 1))/H(ξ, k)^2*φy_j -
                 (H(ξ, k)*∂(f, ξ, k, 2) - f(ξ, k)*∂(H, ξ, k, 2))/H(ξ, k)^2*φx_j)*φ_i*J.dets[k]
    end

    # stamp
    A = Tuple{Int64,Int64,Float64}[]
    for k=1:g.nt, i=1:el.n, j=1:el.n
        if g.t[k, i] ∉ bdy 
            push!(A, (g.t[k, i], g.t[k, j], nuPGCM.ref_el_quad(ξ -> ∫K(ξ, i, j, k), el)))# +
                                            # nuPGCM.ref_el_quad(ξ -> ∫C(ξ, i, j, k), el)))
        end
    end

    # boundary nodes 
    for i ∈ bdy
        push!(A, (i, i, 1))
    end

    # sparse matrix
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N, N)

    return A
end

function build_barotropic_RHS(τx, τy, H, ρ)
    # unpack
    g = τx.g
    bdy = g.e["bdy"]
    J = g.J
    el = g.el
    N = g.np

    # stamp
    rhs = zeros(N)
    for k ∈ 1:g.nt
        function func_r(ξ, i)
            τ_curl = 1/ρ*((H(ξ, k)*∂(τy, ξ, k, 1) - τy(ξ, k)*∂(H, ξ, k, 1)) -
                          (H(ξ, k)*∂(τx, ξ, k, 2) - τx(ξ, k)*∂(H, ξ, k, 2)))/H(ξ, k)^2
            φ_i = φ(el, ξ, i)
            return τ_curl*φ_i*J.dets[k]
        end
        r = [nuPGCM.ref_el_quad(ξ -> func_r(ξ, i), el) for i ∈ 1:el.n]

        rhs[g.t[k, :]] += r
    end

    # boundary nodes 
    for i ∈ bdy
        rhs[i] = 0
    end

    return rhs
end

g = Grid(Triangle(order=1), "../meshes/ocean2.h5")
H = FEField(4e3, g)
r = FEField(1e-5, g)
ϕ(x) = atan(x[3]/sqrt(x[1]^2 + x[2]^2))
θ(x) = atan(x[2], x[1])
τx = FEField(x -> -0.1*cos(6/2*ϕ(x)), g)
τy = FEField(0, g)
ρ = 1000
LHS = build_barotropic_LHS(r, H)
M = nuPGCM.mass_matrix(g)
RHS = M*ones(g.np)
RHS[g.e["bdy"]] .= 0
# RHS = build_barotropic_RHS(τx, τy, H, ρ)
Ψ = LHS\RHS

p = g.p
np = g.np
t = g.t
nt = g.nt
e = g.e["bdy"]
i1 = findall(i -> p[i, 1] ≥ 0, 1:np)
i2 = findall(i -> p[i, 1] ≤ 0, 1:np)
t1 = t[findall(k -> t[k, 1] ∈ i1 && t[k, 2] ∈ i1 && t[k, 3] ∈ i1, 1:nt), :]
e1 = e[findall(i -> e[i] ∈ i1, 1:size(e, 1))]
t2 = t[findall(k -> t[k, 1] ∈ i2 && t[k, 2] ∈ i2 && t[k, 3] ∈ i2, 1:nt), :]
e2 = e[findall(i -> e[i] ∈ i2, 1:size(e, 1))]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2))
nuPGCM.tplot(p[:, 2:3], t1, Ψ; fig, ax=ax[1], contour=true)
ax[1].plot(p[e1, 2], p[e1, 3], "ko", ms=0.3, markeredgecolor="none")
# nuPGCM.tplot(p, t1, τ.values; fig, ax=ax[1])
p_flip = copy(p)
p_flip[:, 2] = -p_flip[:, 2]
nuPGCM.tplot(p_flip[:, 2:3], t2, Ψ; fig, ax=ax[2], contour=true)
ax[2].plot(p_flip[e2, 2], p_flip[e2, 3], "ko", ms=0.3, markeredgecolor="none")
# nuPGCM.tplot(p_flip, t2, τ.values; fig, ax=ax[2])
ax[1].axis("equal")
ax[2].axis("equal")
# ax[1].set_xticks([])
# ax[1].set_yticks([])
# ax[2].set_xticks([])
# ax[2].set_yticks([])
savefig("$out_folder/images/psi.png")
println("$out_folder/images/psi.png")

θs = [θ(p[i, :])*180/π for i ∈ 1:np]
ϕs = [ϕ(p[i, :])*180/π for i ∈ 1:np]
pθϕ = hcat(θs, ϕs)
tθϕ = copy(t)
println("Number of triangles: $nt")
for i ∈ 1:nt
    if !(sign(θs[t[i, 1]]) == sign(θs[t[i, 2]]) == sign(θs[t[i, 3]])) && abs(θs[t[i, 1]]) > 170
        tθϕ[i, :] = t[1, :]
    end
end
tθϕ = unique(tθϕ, dims=1)
ntθϕ = size(tθϕ, 1)
println("Number of triangles: $ntθϕ (after removing θ = 0 crossings)")
# fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.PlateCarree(central_longitude=180)))
# fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.PlateCarree()))
fig, ax = plt.subplots(1)
nuPGCM.tplot(pθϕ, tθϕ, Ψ; contour=true, fig, ax)
# ax.coastlines(lw=0.5)
ax.plot(pθϕ[e, 1], pθϕ[e, 2], "ko", ms=0.3, markeredgecolor="none")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_xticks(-180:90:180)
ax.set_yticks(-90:45:90)
savefig("$out_folder/images/psi_latlon.png")
println("$out_folder/images/psi_latlon.png")