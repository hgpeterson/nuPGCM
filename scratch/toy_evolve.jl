using nuPGCM
using PyPlot
using LinearAlgebra
using Random
using HDF5
using SparseArrays
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output/delta2.5")
if !isdir("$out_folder/data")
    mkdir("$out_folder/data")
end
if !isdir("$out_folder/images")
    mkdir("$out_folder/images")
end

Random.seed!(42)

"""
q = ∇²ψ + βy - ψ/λ²
"""
function build_inversion_LHS(g, K, M, λ)
    A = -K - M/λ^2
    for i ∈ g.e["bdy"]
        A[i, :] .= 0
        A[i, i] = 1.
    end
    return lu(A)
end

function invert!(ψ, LHS, M, q)
    RHS = M*q.values
    RHS[ψ.g.e["bdy"]] .= 0
    ψ.values[:] = LHS\RHS
    return ψ
end

"""
u = -∂y(ψ)
v =  ∂x(ψ)
"""
function compute_velocities(ψ)
    u = -∂(ψ, 2)
    v = ∂(ψ, 1)
    return FVField(u), FVField(v)
end

"""
"""
function build_advection_arrays(g, δ)
    # unpack
    el = g.el
    w = el.quad_wts
    φ = g.φ_qp
    φx = g.∂φ_qp[:, :, 1, :]
    φy = g.∂φ_qp[:, :, 2, :]
    Δ = g.J.dets

    # allocate
    A1 = zeros(g.nt, el.n, el.n, el.n)
    A2 = zeros(g.nt, el.n, el.n, el.n, el.n)
    A3 = zeros(g.nt, el.n, el.n, el.n)

    for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ1 ∈ 1:g.nn, iq ∈ 1:g.nn, i_quad ∈ eachindex(w)
        u1 = -φy[k, iψ1, i_quad]
        v1 =  φx[k, iψ1, i_quad]
        A1[k, i, iψ1, iq] +=      w[i_quad]*(u1*φx[k, iq, i_quad] + v1*φy[k, iq, i_quad])*φ[i, i_quad]*Δ[k]
        A3[k, i, iψ1, iq] += δ[k]*w[i_quad]*(u1*φx[k, i,  i_quad] + v1*φy[k, i,  i_quad])*φ[iq, i_quad]*Δ[k]
        for iψ2 ∈ 1:g.nn
            u2 = -φy[k, iψ2, i_quad]
            v2 =  φx[k, iψ2, i_quad]
            A2[k, i, iψ1, iψ2, iq] += δ[k]*w[i_quad]*(u1*φx[k, iq, i_quad] + v1*φy[k, iq, i_quad])*
                                                     (u2*φx[k, i,  i_quad] + v2*φy[k, i,  i_quad])*Δ[k]
        end
    end

    return A1, A2, A3
end

function advection(A1, A2, q, ψ)
    g = ψ.g
    adv = zeros(g.np)
    for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ1 ∈ 1:g.nn, iq ∈ 1:g.nn
        adv[g.t[k, i]] += A1[k, i, iψ1, iq]*ψ[g.t[k, iψ1]]*q[g.t[k, iq]]
        for iψ2 ∈ 1:g.nn
            adv[g.t[k, i]] += A2[k, i, iψ1, iψ2, iq]*ψ[g.t[k, iψ1]]*ψ[g.t[k, iψ2]]*q[g.t[k, iq]]
        end
    end
    return adv
end

function build_M_SD(A3, q, ψ)
    g = ψ.g
    M_SD_I = zeros(Int64,   g.nt*g.nn^3)
    M_SD_J = zeros(Int64,   g.nt*g.nn^3)
    M_SD_V = zeros(Float64, g.nt*g.nn^3)
    n = 1
    for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ ∈ 1:g.nn, iq ∈ 1:g.nn
        M_SD_I[n] = g.t[k, i]
        M_SD_J[n] = g.t[k, iq]
        M_SD_V[n] = A3[k, i, iψ, iq]*ψ[g.t[k, iψ]]*q[g.t[k, iq]]
        n += 1
    end
    return sparse(M_SD_I, M_SD_J, M_SD_V)
end

function save_q(q, filename)
    file = h5open(filename, "w")
    write(file, "q", q.values)
    close(file)
    println(filename)
end

function read_q(filename)
    file = h5open(filename, "r")
    q = read(file, "q")
    close(file)
    return q
end

function evolve()
    # params
    Δt = 5e-2
    λ = 0.05

    # grid
    # g = Grid(Triangle(order=1), "../meshes/circle/mesh4.h5")
    g = Grid(Triangle(order=1), "../meshes/H/mesh4.h5")

    # δ = const*(local mesh width)
    h = sqrt.(g.J.dets)*2/3^(1/4)
    δ = 2.5*h

    # matrices
    K = nuPGCM.stiffness_matrix(g)
    M = nuPGCM.mass_matrix(g)
    Pinv = sparse(inv(Diagonal(M)))
    inv_LHS = build_inversion_LHS(g, K, M, λ)
    A1, A2, A3 = build_advection_arrays(g, δ)

    # initial condition
    x = g.p[:, 1]
    y = g.p[:, 2]
    # Δ = 0.1
    # q = @. exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) - exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2))
    # q = @. sin(10*π*y)*sin(10*π*x)
    # q .+= 0.05*(rand(g.np) .- 0.5)
    # i_img = 0
    i_img = 195
    q = read_q(@sprintf("%s/data/q%03d.h5", out_folder, i_img))
    qmax = 1.0
    q = FEField(q, g)
    ψ = FEField(0, g)
    invert!(ψ, inv_LHS, M, q)
    quick_plot(q, filename=@sprintf("%s/images/q%03d.png", out_folder, i_img), vmax=qmax)
    # save_q(q, @sprintf("%s/data/q%03d.h5", out_folder, i_img))
    i_img += 1

    # step forward
    t1 = time()
    N = 100000
    dq = zeros(g.np)
    dq_prev = zeros(g.np)
    for i ∈ 1:N
        # update flow
        invert!(ψ, inv_LHS, M, q)

        # compute dq
        adv = advection(A1, A2, q.values, ψ)
        M_SD = build_M_SD(A3, q.values, ψ)
        nuPGCM.cg!(dq, M + M_SD, adv; Pinv)

        if i == 1
            # euler first step
            q.values[:] = q.values - Δt*dq
        else
            # AB2 otherwise
            q.values[:] = q.values - 3/2*Δt*dq + 1/2*Δt*dq_prev
        end

        # save for next step
        dq_prev[:] = dq[:]

        if mod(i, 100) == 0
            # CFL
            u, v = compute_velocities(ψ)
            println("\ni = $i/$N")
            println("CFL Δt: ", min(minimum(abs.(h./u.values)), minimum(abs.(h./v.values))))

            # plot and save
            quick_plot(q, filename=@sprintf("%s/images/q%03d.png", out_folder, i_img), vmax=qmax)
            save_q(q, @sprintf("%s/data/q%03d.h5", out_folder, i_img))
            i_img += 1
        end
    end
    t2 = time()
    println((t2 - t1)/N)

    return q
end

function quick_plot(u; filename, vmax=0)
    g = u.g
    fig, ax = plt.subplots(1, figsize=(3.2, 4.5))
    if vmax == 0
        vmax = maximum(abs(u))
    end
    im = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=true)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("equal")
    savefig(filename, transparent=true)
    println(filename)
    plt.close()
end

q = evolve()

println("Done.")