using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf
using HDF5

set_out_folder("../output")

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

"""
Nonhydrostatic PG equations in TF coordinates:
    -ε²∂σσ(u)/H² - fv + ∂ξ(p) - σHₓ∂σ(p)/H = 0 
    -ε²∂σσ(v)/H² + fu                      = 0
   -γε²∂σσ(w)/H²      + ∂σ(p)/H            = b
                           ∂ξ(Hu) + ∂σ(Hw) = 0
with extra condition
    ∫ p dξ dσ = 0.
Boundary conditions are 
            u = v = w = 0 at σ = -1
    ∂σ(u) = ∂σ(v) = w = 0 at σ = 0 
Weak form:
    ∫ [ε²∂σ(u)∂σ(u₁)/H² - fvu₁ + [∂ξ(p) - σHₓ∂σ(p)/H]u₁ +
       ε²∂σ(v)∂σ(u₂)/H² + fuu₂  
      γε²∂σ(w)∂σ(u₃)/H²        + ∂σ(p)u₃/H +
                ∂ξ(Hu)q + ∂σ(Hw)q
      ] dξ dσ
    = ∫ bu₃ dξ dσ
for all 
    u₁, u₂, u₃ ∈ P₂ and q ∈ P₁,
where Pₙ is the space of continuous polynomials of degree n.
"""
function build_nonhydro(ε², γ, f, H, Hx, g1, g2, b) 
    # unpack
    J = g1.J 
    el1 = g1.el
    el2 = g2.el
    w = el1.quad_wts
    qp = el1.quad_pts
    φ1 = g1.φ_qp
    φ2 = g2.φ_qp
    ∂φ1 = g1.∂φ_qp
    ∂φ2 = g2.∂φ_qp
    sfc = g2.e["sfc"]
    bot = g2.e["bot"]
    coast = g2.e["coast"]
    bdy = unique([sfc; bot; coast])
    bot_coast = unique([bot; coast])

    # indices
    umap = reshape(1:3*g2.np, (3, g2.np))
    pmap = umap[end] .+ (1:g1.np)
    N = pmap[end]
    println("N = $N")
    i_pbc = 1

    # stamp system
    A = Tuple{Int64,Int64,Float64}[]
    r = zeros(N)
    for k ∈ 1:g1.nt
        x_qp = [transform_from_ref_el(el1, qp[i_quad, :], g1.p[g1.t[k, :], :])[1] for i_quad ∈ eachindex(w)]
        σ_qp = [transform_from_ref_el(el1, qp[i_quad, :], g1.p[g1.t[k, :], :])[2] for i_quad ∈ eachindex(w)]
        for i ∈ 1:el2.n
            for j ∈ 1:el2.n
                # ∫ ∂σ(φᵢ)∂σ(φ₂)/H²
                K = dot(w, ∂φ2[k, i, 2, :].*∂φ2[k, j, 2, :]./H.(x_qp).^2)*J.dets[k]
                if g2.t[k, i] ∉ bot_coast
                    # ε²∂z(u)∂z(u₁)
                    push!(A, (umap[1, g2.t[k, i]], umap[1, g2.t[k, j]], ε²*K))
                    # ε²∂z(v)∂z(u₂)
                    push!(A, (umap[2, g2.t[k, i]], umap[2, g2.t[k, j]], ε²*K))
                end
                if g2.t[k, i] ∉ bdy
                    # γε²∂z(w)∂z(u₃)
                    push!(A, (umap[3, g2.t[k, i]], umap[3, g2.t[k, j]], γ*ε²*K))
                end

                # ∫ φᵢφ₂ 
                M = dot(w, φ2[i, :].*φ2[j, :])*J.dets[k]
                if g2.t[k, i] ∉ bot_coast
                    # -fvu₁
                    push!(A, (umap[2, g2.t[k, i]], umap[1, g2.t[k, j]], -f*M))
                    # fuu₂
                    push!(A, (umap[1, g2.t[k, i]], umap[2, g2.t[k, j]], f*M))
                    # bu₃
                    r[umap[3, g2.t[k, i]]] += M*b[g2.t[k, j]] 
                end
            end
            for j ∈ 1:el1.n
                # ∫ φᵢ[∂ξ(φⱼ) - σHₓ∂σ(φⱼ)/H]
                Cξ = dot(w, φ2[i, :].*(∂φ1[k, j, 1, :] - σ_qp.*Hx.(x_qp).*∂φ1[k, j, 2, :]./H.(x_qp)))*J.dets[k]
                if g2.t[k, i] ∉ bot_coast
                    # u₁[∂ξ(p) - σHₓ∂σ(φⱼ)/H]
                    push!(A, (umap[1, g2.t[k, i]], pmap[g1.t[k, j]], Cξ))
                end
                # ∫ φᵢ∂σ(φⱼ)/H 
                Cσ = dot(w, φ2[i, :].*∂φ1[k, j, 2, :]./H.(x_qp))*J.dets[k]
                # u₃∂z(p)
                if g2.t[k, i] ∉ bdy
                    push!(A, (umap[3, g2.t[k, i]], pmap[g1.t[k, j]], Cσ))
                end

                if g1.t[k, j] !== i_pbc
                    # ∫ φᵢ∂ξ(Hφⱼ) 
                    Cξ = dot(w, φ1[j, :].*(Hx.(x_qp).*φ2[i, :] .+ H.(x_qp).*∂φ2[k, i, 1, :]))*J.dets[k]
                    # q∂ξ(Hu)
                    push!(A, (pmap[g1.t[k, j]], umap[1, g2.t[k, i]], Cξ))
                    # ∫ φᵢ∂σ(Hφⱼ) 
                    Cσ = dot(w, φ1[j, :].*H.(x_qp).*∂φ2[k, i, 2, :])*J.dets[k]
                    # q∂σ(Hw)
                    push!(A, (pmap[g1.t[k, j]], umap[3, g2.t[k, i]], Cσ))
                end
            end
        end
    end

    # for i ∈ bdy, j ∈ 1:3
    #     push!(A, (umap[j, i], umap[j, i], 1))
    # end
    # r[umap[:, bdy]] .= 0

    # u = v = w = 0 at σ = -1 and coast
    for i ∈ bot_coast, j ∈ 1:3
        push!(A, (umap[j, i], umap[j, i], 1))
    end
    r[umap[:, bot_coast]] .= 0

    # ∂σ(u) = ∂σ(v) = 0 at σ = 0 → natural

    # w = 0 at σ = 0
    for i ∈ sfc
        push!(A, (umap[3, i], umap[3, i], 1))
    end
    r[umap[3, sfc]] .= 0

    # set ∑p to zero
    for i ∈ 1:g1.np
        push!(A, (pmap[i_pbc], pmap[i], 1))
    end
    # # set p[i_pbc] to zero
    # push!(A, (pmap[i_pbc], pmap[i_pbc], 1))
    r[pmap[i_pbc]] = 0

    # make CSC matrix
    A = sparse((x -> x[1]).(A), (x -> x[2]).(A), (x -> x[3]).(A), N, N)

    # println(N)
    # println(rank(A))

    return A, r, umap, pmap
end

function solve_nonhydro()
    # params
    ε² = 1e-2
    γ = 1/8
    f = 1

    # depth function 
    H(x) = 1 - x^2
    Hx(x) = -2x

    # grids
    nσ = 2^6
    nξ = Int64(round(nσ/γ))
    println("nσ = $nσ")
    println("nξ = $nξ")
    # σ = -(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2
    # ξ = -cos.(π*(0:nξ-1)/(nξ-1))
    ξ = range(-1, 1, length=nξ)
    σ = range(-1, 0, length=nσ)
    println("σ[2] - σ[1] = $(σ[2] - σ[1])")
    println("          δ = $(sqrt(2ε²))")
    p = zeros(nξ*nσ, 2)
    t = zeros(Int64, 2(nξ - 1)*(nσ - 1), 3)
    e = zeros(Int64, 2nξ + 2nσ - 4)
    bot = ((1:nξ) .- 1)*nσ .+ 1
    sfc = (1:nξ)*nσ
    left = 2:nσ - 1
    right = (nξ - 1)*nσ .+ (2:nσ - 1)
    coast = [left; right; bot[1]; bot[end]; sfc[1]; sfc[end]]
    e[1:nξ] = bot
    e[nξ + 1:2nξ] = sfc 
    e[2nξ + 1:2nξ + nσ - 2] = left
    e[2nξ + nσ - 1:2nξ + 2nσ - 4] = right
    for i ∈ 1:nξ
        for j ∈ 1:nσ
            p[(i - 1)*nσ + j, 1] = ξ[i]
            p[(i - 1)*nσ + j, 2] = σ[j]
            if i < nξ && j < nσ
                t[2*((i - 1)*(nσ - 1) + j) - 1, 1] = (i - 1)*nσ + j
                t[2*((i - 1)*(nσ - 1) + j) - 1, 2] = i*nσ + j
                t[2*((i - 1)*(nσ - 1) + j) - 1, 3] = i*nσ + j + 1
                t[2*((i - 1)*(nσ - 1) + j), 1] = (i - 1)*nσ + j
                t[2*((i - 1)*(nσ - 1) + j), 2] = i*nσ + j + 1
                t[2*((i - 1)*(nσ - 1) + j), 3] = (i - 1)*nσ + j + 1
            end
        end
    end
    # file = h5open("../meshes/rectangle/mesh4.h5", "r")
    # p = read(file, "p")
    # t = read(file, "t")
    # e = read(file, "e")
    # close(file)
    # p[:, 1] = 2*p[:, 1] .- 1
    # p[:, 2] = (p[:, 2] .- 1)/2
    # sfc = e[abs.(p[e, 2]) .< 1e-4]
    # bot = e[abs.(p[e, 2] .+ 1) .< 1e-4]
    # left = e[abs.(p[e, 1] .+ 1) .< 1e-4]
    # right = e[abs.(p[e, 1] .- 1) .< 1e-4]
    # coast = unique([left; right])
    g1 = Grid(Triangle(order=1), p, t, Dict("sfc" => sfc, "bot" => bot, "coast" => coast))
    g2 = add_midpoints(g1)
    fig, ax, im = nuPGCM.tplot(g2.p, g2.t)
    sfc2 = g2.e["sfc"]
    bot2 = g2.e["bot"]
    coast2 = g2.e["coast"]
    ax.plot(g2.p[coast2, 1], g2.p[coast2, 2], "go", ms=1, markeredgecolor="none")
    ax.plot(g2.p[sfc2, 1],   g2.p[sfc2, 2],   "ro", ms=1, markeredgecolor="none")
    ax.plot(g2.p[bot2, 1],   g2.p[bot2, 2],   "bo", ms=1, markeredgecolor="none")
    ax.axis("equal")
    savefig("$out_folder/images/mesh.png")
    println("$out_folder/images/mesh.png")
    plt.close()

    # buoyancy
    x2 = g2.p[:, 1]
    z2 = g2.p[:, 2].*H.(x2)
    # b = @. -exp(-x2^2/0.1 - (z2 + 0.5)^2/0.1)
    b = @. z2 + 0.1*exp(-(z2 + H(x2))/0.1)
    quick_plot(FEField(b, g2), FEField(b, g2), H, L"b", "$out_folder/images/b.png")

    # solve
    @time A, r, umap, pmap = build_nonhydro(ε², γ, f, H, Hx, g1, g2, b) 
    @time sol = A\r
    # sol = zeros(size(r))
    # @time nuPGCM.cg!(sol, A, r)
    u = sol[umap[1, :]]
    v = sol[umap[2, :]]
    w = sol[umap[3, :]]
    p = sol[pmap]

    quick_plot(FEField(u, g2), FEField(b, g2), H, L"u", "$out_folder/images/u.png")
    quick_plot(FEField(v, g2), FEField(b, g2), H, L"v", "$out_folder/images/v.png")
    quick_plot(FEField(w, g2), FEField(b, g2), H, L"w", "$out_folder/images/w.png")
    quick_plot(FEField(p, g1), FEField(b, g2), H, L"p", "$out_folder/images/p.png")
end

function quick_plot(u, b, H, label, filename)
    vmax = maximum(abs(u))
    fig, ax = plt.subplots(1)
    img = ax.tripcolor(u.g.p[:, 1], u.g.p[:, 2].*H.(u.g.p[:, 1]), u.g.t[:, 1:3] .- 1, u.values, shading="gouraud", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
    plt.colorbar(img, ax=ax, label=label)
    ax.tricontour(b.g.p[:, 1], b.g.p[:, 2].*H.(b.g.p[:, 1]), b.g.t[:, 1:3] .- 1, b.values, colors="k", linestyles="-", linewidths=0.5, levels=20, alpha=0.25)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.axis("equal")
    ax.set_xlabel(L"x")
    ax.set_ylabel(L"z")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    savefig(filename)
    println(filename)
    plt.close()
end

solve_nonhydro()