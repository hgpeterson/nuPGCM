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
        φ_i = φ(g.el, ξ, i)
        ∇φ_i = φξ(el, ξ, i)*J.Js[k, 1, :] + φη(el, ξ, i)*J.Js[k, 2, :]
        ∇φ_j = φξ(el, ξ, j)*J.Js[k, 1, :] + φη(el, ξ, j)*J.Js[k, 2, :]
        ∇H = [∂(H, ξ, k, 1), ∂(H, ξ, k, 2)]
        # return -r(ξ, k)/H(ξ, k)*dot(∇φ_i, ∇φ_j)*J.dets[k]
        return -2*r(ξ, k)*dot(∇φ_j, ∇H)*φ_i*J.dets[k] - r(ξ, k)*H(ξ, k)*dot(∇φ_i, ∇φ_j)*J.dets[k]
    end
    function ∫C(ξ, i, j, k)
        # ∇φ_j = φξ(el, ξ, j)*J.Js[k, 1, :] + φη(el, ξ, j)*J.Js[k, 2, :]
        φx_j = φξ(el, ξ, j)*J.Js[k, 1, 1] + φη(el, ξ, j)*J.Js[k, 2, 1]
        φy_j = φξ(el, ξ, j)*J.Js[k, 1, 2] + φη(el, ξ, j)*J.Js[k, 2, 2]
        φ_i = φ(g.el, ξ, i)
        # return -((H(ξ, k)*∂(f, ξ, k, 1) - f(ξ, k)*∂(H, ξ, k, 1))/H(ξ, k)^2*φy_j -
        #          (H(ξ, k)*∂(f, ξ, k, 2) - f(ξ, k)*∂(H, ξ, k, 2))/H(ξ, k)^2*φx_j)*φ_i*J.dets[k]
        return -((H(ξ, k)*∂(f, ξ, k, 1) - f(ξ, k)*∂(H, ξ, k, 1))*φy_j -
                 (H(ξ, k)*∂(f, ξ, k, 2) - f(ξ, k)*∂(H, ξ, k, 2))*φx_j)*φ_i*J.dets[k]
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
    # A[bdy[1], :] .= 0
    # A[bdy[1], bdy[1]] = 1
    # println(rank(A))
    # println(N)

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

        # local triangle coordinates (e_x′, e_y′)
        v1 = p2 - p1
        v2 = p3 - p1
        e_x′ = v1/norm(v1)
        e_y′ = cross(cross(v1, v2), v1)
        e_y′ /= norm(e_y′)

        # zonal/meridional coordinates (e_x, e_y)
        λ0 = λ(p1)
        ϕ0 = ϕ(p1)
        e_x = [-sin(λ0), cos(λ0), 0]
        e_y = [-cos(λ0)*sin(ϕ0), -sin(λ0)*sin(ϕ0), cos(ϕ0)]

        # transform from zonal/meridional to local coordinates
        τx′ = dot(e_x, e_x′)*τx + dot(e_y, e_x′)*τy
        τy′ = dot(e_x, e_y′)*τx + dot(e_y, e_y′)*τy
        function func_r(ξ, i)
            # τ_curl = 1/ρ*((H(ξ, k)*∂(τy′, ξ, k, 1) - τy′(ξ, k)*∂(H, ξ, k, 1)) -
            #               (H(ξ, k)*∂(τx′, ξ, k, 2) - τx′(ξ, k)*∂(H, ξ, k, 2)))/H(ξ, k)^2
            τ_curl = 1/ρ*((H(ξ, k)*∂(τy′, ξ, k, 1) - τy′(ξ, k)*∂(H, ξ, k, 1)) -
                          (H(ξ, k)*∂(τx′, ξ, k, 2) - τx′(ξ, k)*∂(H, ξ, k, 2)))
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
    # rhs[bdy[1]] = 0

    return rhs
end

function RobinsonProj(ϕ, λ)
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
    return 0.8487*R*X*(λ)*π/180, 1.3523*R*Y*hemisphere
end

# grid
# g = Grid(Triangle(order=1), "../meshes/ocean.h5")
g = Grid(Triangle(order=1), "../../oceanmesh/ocean.h5")
p = g.p
np = g.np
t = g.t
nt = g.nt
e = g.e["bdy"]

# plotting grid
R = 6.371e6
ϕ(x) = atan(x[3]/sqrt(x[1]^2 + x[2]^2))
λ(x) = atan(x[2], x[1])
ϕs = [ϕ(p[i, :])*180/π for i ∈ 1:np]
λs = [λ(p[i, :])*180/π for i ∈ 1:np]
p_latlon = hcat(ϕs, λs)
p_robinson = zeros(np, 2)
for i ∈ 1:np
    x, y = RobinsonProj(ϕs[i], λs[i])
    p_robinson[i, :] = [x, y]
end
good_tris = findall(i -> (sign(λs[t[i, 1]]) == sign(λs[t[i, 2]]) == sign(λs[t[i, 3]])) || abs(λs[t[i, 1]]) < 10, 1:nt)
t_latlon = t[good_tris, :]
nt_latlon = size(t_latlon, 1)
println("Number of triangles: $nt")
println("Number of triangles: $nt_latlon (after removing λ = ±180 crossings)")

# # plot ϕ
# fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
# nuPGCM.tplot(p_robinson, t_latlon, ϕs; fig, ax, contour=true, contour_levels=10, cb_orientation="horizontal", cb_label=L"$\phi$ (°)")
# ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
# savefig("$out_folder/images/phi.png")
# println("$out_folder/images/phi.png")

# # plot λ
# fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
# nuPGCM.tplot(p_robinson, t_latlon, λs; fig, ax, contour=true, contour_levels=10, cb_orientation="horizontal", cb_label=L"$\lambda$ (°)")
# ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
# savefig("$out_folder/images/lambda.png")
# println("$out_folder/images/lambda.png")

# mesh
if np < 1e6
    fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
    nuPGCM.tplot(p_robinson, t_latlon; fig, ax)
    ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
    savefig("$out_folder/images/mesh.png")
    println("$out_folder/images/mesh.png")
end

# depth
H = FEField(4e3, g)
# H = h5open("H.h5", "r") do file
#     read(file, "H")
# end
# H .+= 10
# H[e] .= 0
# H = FEField(H, g)
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, t_latlon, H.values; fig, ax, cb_orientation="horizontal", cb_label=L"$H$ (m)", cmap="Blues", vmin=0, vmax=1e4)
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
ax.set_xticks([])
ax.set_yticks([])
savefig("$out_folder/images/H.png")
println("$out_folder/images/H.png")

# drag
r = FEField(1e-5, g)
println(@sprintf("Stommel BL width: %d km", r[1]/(2*7.272e-5)*6.371e6/1000))
areas = nuPGCM.tri_areas(g)
h = sqrt.(4*areas/π)
println(@sprintf("Minimum resolution: %d km", minimum(h)/1000))
println(@sprintf("Maximum resolution: %d km", maximum(h)/1000))

# wind
τx = FEField(x -> -0.3*cos(5*ϕ(x)), g)
τy = FEField(0, g)
ρ = 1000
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, t_latlon, τx.values; fig, ax, cb_orientation="horizontal", cb_label=L"$\tau^x$ (N m$^{-2}$)")
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
savefig("$out_folder/images/tau.png")
println("$out_folder/images/tau.png")

# solve
LHS = build_barotropic_LHS(r, H)
RHS = build_barotropic_RHS(τx, τy, H, ρ)
fig, ax = plt.subplots(subplot_kw=Dict("projection"=>crs.Robinson()))
nuPGCM.tplot(p_robinson, t_latlon, RHS; fig, ax, cb_orientation="horizontal", cb_label="RHS")
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
nuPGCM.tplot(p_robinson, t_latlon, Ψ/1e6; contour=true, contour_levels=10, fig, ax, cb_orientation="horizontal", cb_label=L"$\Psi$ (Sv)")
ax.plot(p_robinson[e, 1], p_robinson[e, 2], "ko", ms=0.3, markeredgecolor="none")
savefig("$out_folder/images/psi_latlon.png")
println("$out_folder/images/psi_latlon.png")

# plot RHS
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2))
nuPGCM.tplot(p[:, 2:3], t1, RHS; fig, ax=ax[1], cb_label="RHS")
ax[1].plot(p[e1, 2], p[e1, 3], "ko", ms=0.3, markeredgecolor="none")
nuPGCM.tplot(p_flip[:, 2:3], t2, RHS; fig, ax=ax[2], cb_label="RHS")
ax[2].plot(p_flip[e2, 2], p_flip[e2, 3], "ko", ms=0.3, markeredgecolor="none")
ax[1].axis("equal")
ax[2].axis("equal")
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].set_xticks([])
ax[2].set_yticks([])
savefig("$out_folder/images/RHS_orthographic.png")
println("$out_folder/images/RHS_orthographic.png")