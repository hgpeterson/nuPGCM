using nuPGCM
using PyPlot
using PyCall
using LinearAlgebra
using SparseArrays
using HDF5
using Printf

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
            push!(A, (g.t[k, i], g.t[k, j], nuPGCM.ref_el_quad(ξ -> ∫K(ξ, i, j, k), el) +
                                            nuPGCM.ref_el_quad(ξ -> ∫C(ξ, i, j, k), el)))
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
        # local tri
        p = g.p[g.t[k, :], :]
        p1 = p[1, :]
        p2 = p[2, :]
        p3 = p[3, :]

        # local triangle coordinates (x′, y′, z′)
        v1 = p2 - p1
        v2 = p3 - p1
        x′ = v1/norm(v1)
        y′ = cross(cross(v1, v2), v1)
        y′ /= norm(y′)

        # zonal/meridional coordinates (x, y, z)
        z = p1/norm(p1)
        θ0 = θ(z)
        ϕ0 = ϕ(z)
        ϕ1 = ϕ0 + 0.1 
        y1 = [sin(ϕ1)*cos(θ0), sin(ϕ1)*sin(θ0), cos(ϕ1)] - z
        x = cross(y1, z)
        x /= norm(x)
        y = cross(z, x)

        # transform from zonal/meridional to local coordinates
        τx′ = dot(x, x′)*τx + dot(y, x′)*τy
        τy′ = dot(x, y′)*τx + dot(y, y′)*τy
        function func_r(ξ, i)
            τ_curl = 1/ρ*((H(ξ, k)*∂(τy′, ξ, k, 1) - τy′(ξ, k)*∂(H, ξ, k, 1)) -
                          (H(ξ, k)*∂(τx′, ξ, k, 2) - τx′(ξ, k)*∂(H, ξ, k, 2)))/H(ξ, k)^2
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

function RobinsonProj(ϕ, θ)
    R = 6.371e6
    ϕs = 0:5:90
    Xs = [1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322]
    Ys = [0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000]
    hemisphere = 1
    if ϕ < 0
        ϕ = -ϕ
        hemisphere = -1
    end
    X = nuPGCM.lerp(ϕs, Xs, ϕ)
    Y = nuPGCM.lerp(ϕs, Ys, ϕ)
    return 0.8487*R*X*(θ)*π/180, 1.3523*R*Y*hemisphere
end

# grid
g = Grid(Triangle(order=1), "../meshes/ocean.h5")
p = g.p
np = g.np
t = g.t
nt = g.nt
e = g.e["bdy"]

# plotting grid
ϕ(x) = atan(x[3]/sqrt(x[1]^2 + x[2]^2))
θ(x) = atan(x[2], x[1])
θs = [θ(p[i, :])*180/π for i ∈ 1:np]
ϕs = [ϕ(p[i, :])*180/π for i ∈ 1:np]
pθϕ = hcat(θs, ϕs)
p_robinson = zeros(np, 2)
for i ∈ 1:np
    x, y = RobinsonProj(ϕs[i], θs[i])
    p_robinson[i, :] = [x, y]
end
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

# mesh
if np < 1e6
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
    nuPGCM.tplot(p_robinson, tθϕ; fig, ax)
    ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
    savefig("$out_folder/images/mesh.png")
    println("$out_folder/images/mesh.png")
end

# depth
# H = FEField(4e3, g)
H = h5open("H.h5", "r") do file
    read(file, "H")
end
H = FEField(H .+ 10, g)
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, tθϕ, H.values; fig, ax, cb_orientation="horizontal", cb_label=L"$H$ (m)", cmap="Blues", vmin=0, vmax=1e4)
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
ax.set_xticks([])
ax.set_yticks([])
savefig("$out_folder/images/H.png")
println("$out_folder/images/H.png")

# drag
r = FEField(5e-6, g)
println(@sprintf("Stommel BL width: %d km", r[1]/(2*7.272e-5)*6.371e6/1000))
println(@sprintf("Minimum resolution: %d km", minimum([norm(p[t[i, 1], :] - p[t[i, 2], :]) for i ∈ 1:nt])/1000))
println(@sprintf("Maximum resolution: %d km", maximum([norm(p[t[i, 1], :] - p[t[i, 2], :]) for i ∈ 1:nt])/1000))

# wind
τx = FEField(x -> -0.3*cos(4*ϕ(x)), g)
τy = FEField(0, g)
ρ = 1000
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, tθϕ, τx.values; fig, ax, cb_orientation="horizontal", cb_label=L"$\tau^x$ (N m$^{-2}$)")
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
savefig("$out_folder/images/tau.png")
println("$out_folder/images/tau.png")

# solve
LHS = build_barotropic_LHS(r, H)
RHS = build_barotropic_RHS(τx, τy, H, ρ)
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, tθϕ, RHS; fig, ax, cb_orientation="horizontal", cb_label="RHS")
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
savefig("$out_folder/images/RHS.png")
println("$out_folder/images/RHS.png")
Ψ = LHS\RHS

# plot Ψ
i1 = findall(i -> p[i, 1] ≥ 0, 1:np)
i2 = findall(i -> p[i, 1] ≤ 0, 1:np)
t1 = t[findall(k -> t[k, 1] ∈ i1 && t[k, 2] ∈ i1 && t[k, 3] ∈ i1, 1:nt), :]
e1 = e[findall(i -> e[i] ∈ i1, 1:size(e, 1))]
t2 = t[findall(k -> t[k, 1] ∈ i2 && t[k, 2] ∈ i2 && t[k, 3] ∈ i2, 1:nt), :]
e2 = e[findall(i -> e[i] ∈ i2, 1:size(e, 1))]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2))
nuPGCM.tplot(p[:, 2:3], t1, Ψ/1e6; fig, ax=ax[1], contour=true, cb_label=L"$\Psi$ (Sv)")
ax[1].plot(p[e1, 2], p[e1, 3], "ko", ms=0.3, markeredgecolor="none")
p_flip = copy(p)
p_flip[:, 2] = -p_flip[:, 2]
nuPGCM.tplot(p_flip[:, 2:3], t2, Ψ/1e6; fig, ax=ax[2], contour=true, cb_label=L"$\Psi$ (Sv)")
ax[2].plot(p_flip[e2, 2], p_flip[e2, 3], "ko", ms=0.3, markeredgecolor="none")
ax[1].axis("equal")
ax[2].axis("equal")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_xticks([])
ax[2].set_yticks([])
savefig("$out_folder/images/psi.png")
println("$out_folder/images/psi.png")

fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, tθϕ, Ψ/1e6; contour=true, contour_levels=10, fig, ax, cb_orientation="horizontal", cb_label=L"$\Psi$ (Sv)")
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
savefig("$out_folder/images/psi_latlon.png")
println("$out_folder/images/psi_latlon.png")