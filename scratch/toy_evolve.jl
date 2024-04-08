using nuPGCM
using PyPlot
using LinearAlgebra
using Random
using HDF5
using SparseArrays
using Printf
using CUDA
using CUDA.CUSPARSE

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

set_out_folder("../output")
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

    return CuArray(A1), CuArray(A2), CuArray(A3)
end

# function advection(A1, A2, q, ψ)
#     g = ψ.g
#     adv = zeros(g.np)
#     for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ1 ∈ 1:g.nn, iq ∈ 1:g.nn
#         adv[g.t[k, i]] += A1[k, i, iψ1, iq]*ψ[g.t[k, iψ1]]*q[g.t[k, iq]]
#         for iψ2 ∈ 1:g.nn
#             adv[g.t[k, i]] += A2[k, i, iψ1, iψ2, iq]*ψ[g.t[k, iψ1]]*ψ[g.t[k, iψ2]]*q[g.t[k, iq]]
#         end
#     end
#     return adv
# end
function advection(A1, A2, q, ψ, t)
# function advection(δ, w, φ, φx, φy, Δ, q, ψ, t)
    # unpack
    g = q.g

    # load arrays on GPU
    adv = CUDA.zeros(g.nt, g.nn) 
    q_gpu = CuArray(q.values)
    ψ_gpu = CuArray(ψ.values)

    # setup advection kernel
    kernel = @cuda launch=false gpu_adv!(adv, A1, A2, q_gpu, ψ_gpu, t)
    # kernel = @cuda launch=false gpu_adv!(adv, δ, w, φ, φx, φy, Δ, q_gpu, ψ_gpu, t)
    config = launch_configuration(kernel.fun)
    threads = min(g.nt, config.threads)
    blocks = cld(g.nt, threads)

    CUDA.@sync begin
        kernel(adv, A1, A2, q_gpu, ψ_gpu, t; threads, blocks)
        # kernel(adv, δ, w, φ, φx, φy, Δ, q_gpu, ψ_gpu, t; threads, blocks)
    end

    # copy result to CPU
    cpu_adv = Array(adv)
    return cpu_adv
end

function gpu_adv!(adv, A1, A2, q, ψ, t)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for k ∈ index:stride:size(A1, 1), i ∈ axes(A1, 2), iψ1 ∈ axes(A1, 3), iq ∈ axes(A1, 4)
        adv[k, i] += A1[k, i, iψ1, iq]*ψ[t[k, iψ1]]*q[t[k, iq]]
        for iψ2 ∈ axes(A2, 4)
            adv[k, i] += A2[k, i, iψ1, iψ2, iq]*ψ[t[k, iψ1]]*ψ[t[k, iψ2]]*q[t[k, iq]]
        end
    end
    return
end

# function gpu_adv!(adv, δ, w, φ, φx, φy, Δ, q, ψ, t)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for k ∈ index:stride:length(Δ), i ∈ axes(t, 2), iψ1 ∈ axes(t, 2), iq ∈ axes(t, 2), i_quad ∈ eachindex(w)
#         adv[k, i] += w[i_quad]*(φx[k, iψ1, i_quad]*φy[k, iq, i_quad] - φy[k, iψ1, i_quad]*φx[k, iq, i_quad])*φ[i, i_quad]*Δ[k]*ψ[t[k, iψ1]]*q[t[k, iq]]
#         for iψ2 ∈ axes(t, 2)
#             adv[k, i] += δ[k]*w[i_quad]*(φx[k, iψ1, i_quad]*φy[k, iq, i_quad] - φy[k, iψ1, i_quad]*φx[k, iq, i_quad])*
#                                         (φx[k, iψ2, i_quad]*φy[k, i,  i_quad] - φy[k, iψ2, i_quad]*φx[k, i,  i_quad])*Δ[k]*ψ[t[k, iψ1]]*ψ[t[k, iψ2]]*q[t[k, iq]]
#         end
#     end
#     return
# end

# function build_M_SD(A3, q, ψ)
#     g = ψ.g
#     M_SD_I = zeros(Int64,   g.nt*g.nn^3)
#     M_SD_J = zeros(Int64,   g.nt*g.nn^3)
#     M_SD_V = zeros(Float64, g.nt*g.nn^3)
#     n = 1
#     for k ∈ 1:g.nt, i ∈ 1:g.nn, iψ ∈ 1:g.nn, iq ∈ 1:g.nn
#         M_SD_I[n] = g.t[k, i]
#         M_SD_J[n] = g.t[k, iq]
#         M_SD_V[n] = A3[k, i, iψ, iq]*ψ[g.t[k, iψ]]*q[g.t[k, iq]]
#         n += 1
#     end
#     return sparse(M_SD_I, M_SD_J, M_SD_V)
# end
function build_M_SD(M_SD_I, M_SD_J, A3, q, ψ, t)
    # unpack
    g = q.g

    # load arrays on GPU
    M_SD_V = CUDA.zeros(g.nt, g.nn, g.nn) 
    ψ_gpu = CuArray(ψ.values)
    q_gpu = CuArray(q.values)

    # setup kernel
    kernel = @cuda launch=false gpu_M_SD!(M_SD_V, A3, q_gpu, ψ_gpu, t)
    config = launch_configuration(kernel.fun)
    threads = min(g.nt, config.threads)
    blocks = cld(g.nt, threads)

    CUDA.@sync begin
        kernel(M_SD_V, A3, q_gpu, ψ_gpu, t; threads, blocks)
    end

    # sparse matrix
    return CuSparseMatrixCSC(M_SD_I, M_SD_J, M_SD_V[:], (g.np, g.np))
end
function build_M_SD_IJ(g)
    M_SD_I = zeros(Int32, g.nt*g.nn*g.nn) 
    M_SD_J = zeros(Int32, g.nt*g.nn*g.nn) 
    n = 1
    for k ∈ 1:g.nt, i ∈ 1:g.nn, iq ∈ 1:g.nn
        M_SD_I[n] = g.t[k, i]
        M_SD_J[n] = g.t[k, iq]
        n += 1
    end
    M_SD_I = CuArray(M_SD_I)
    M_SD_J = CuArray(M_SD_J)
    return M_SD_I, M_SD_J
end
function gpu_M_SD!(M_SD_V, A3, q, ψ, t)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for k ∈ index:stride:size(A3, 1), i ∈ axes(A3, 2), iψ ∈ axes(A3, 3), iq ∈ axes(A3, 4)
        M_SD_V[k, i, iq] += A3[k, i, iψ, iq]*ψ[t[k, iψ]]*q[t[k, iq]]
    end
    return 
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
    g = Grid(Triangle(order=1), "../meshes/H/mesh5.h5")
    # g = add_midpoints(g)
    println("DoF: $(g.np)")
    t_gpu = CuArray(g.t)
    el_map = nuPGCM.build_element_map(g)

    # δ = const*(local mesh width)
    h = sqrt.(4/π*g.J.dets) # diameter of circle with element area
    δ = 4*h

    # δ_gpu = CuArray(δ)
    # el = g.el
    # w_gpu = CuArray(el.quad_wts)
    # φ_gpu = CuArray(g.φ_qp)
    # φx_gpu = CuArray(g.∂φ_qp[:, :, 1, :])
    # φy_gpu = CuArray(g.∂φ_qp[:, :, 2, :])
    # Δ_gpu = CuArray(g.J.dets)

    # matrices
    K = nuPGCM.stiffness_matrix(g)
    M = nuPGCM.mass_matrix(g)
    M_gpu = CuSparseMatrixCSC(M)
    M_SD_I, M_SD_J = build_M_SD_IJ(g)
    Pinv = CuSparseMatrixCSC(sparse(inv(Diagonal(M))))
    inv_LHS = build_inversion_LHS(g, K, M, λ)
    A1, A2, A3 = build_advection_arrays(g, δ)

    CUDA.memory_status()

    # initial condition
    x = g.p[:, 1]
    y = g.p[:, 2]
    # Δ = 0.1
    # q = @. exp(-(x + 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2)) - exp(-(x - 0.25)^2/(2*Δ^2) - y^2/(2*Δ^2))
    F = @. sin(10*π*y)*sin(10*π*x)
    τ = 1e2
    q = F .+ 0.05*(rand(g.np) .- 0.5)
    i_img = 0
    # q = read_q(@sprintf("%s/data/q%03d.h5", out_folder, i_img))
    qmax = 1.0
    q = FEField(q, g)
    ψ = FEField(0, g)
    invert!(ψ, inv_LHS, M, q)
    quick_plot(q, filename=@sprintf("%s/images/q%03d.png", out_folder, i_img), vmax=qmax)
    save_q(q, @sprintf("%s/data/q%03d.h5", out_folder, i_img))
    i_img += 1

    # step forward
    t1 = time()
    N = 10000
    dq = CUDA.zeros(eltype(M_gpu), g.np) # pre-allocate for `cg!`
    dq_prev = zeros(eltype(M_gpu), g.np) 
    for i ∈ 1:N
        # update flow
        invert!(ψ, inv_LHS, M, q)

        # compute dq
        # @time "adv" begin # 0.004
            adv_el = advection(A1, A2, q, ψ, t_gpu)
            # adv_el = advection(δ_gpu, w_gpu, φ_gpu, φx_gpu, φy_gpu, Δ_gpu, q, ψ, t_gpu)
            adv = CuArray(el_map*adv_el[:])
            M_SD = build_M_SD(M_SD_I, M_SD_J, A3, q, ψ, t_gpu)
        # end
        # @time "cg!" begin
            nuPGCM.cg!(dq, M_gpu + M_SD, adv; Pinv) # 0.1
        # end

        if i == 1
            # euler first step
            q.values[:] = q.values - Δt*Array(dq) + Δt*(F - q.values)/τ
        else
            # AB2 otherwise
            q.values[:] = q.values - 3/2*Δt*Array(dq) + 1/2*Δt*dq_prev + Δt*(F - q.values)/τ
        end

        # save for next step
        dq_prev[:] = Array(dq)[:]

        if mod(i, 100) == 0
            elapsed = time() - t1
            elapsed_h, elapsed_m, elapsed_s = hrs_mins_secs(elapsed)
            ETR = (N - i)*elapsed/i
            ETR_h, ETR_m, ETR_s = hrs_mins_secs(ETR)
            @printf("\nt = %.2e (%d/%d steps)\n", i*Δt, i, N)
            @printf("    time elapsed:   %02d:%02d:%02d\n", elapsed_h, elapsed_m, elapsed_s)
            @printf("    time remaining: %02d:%02d:%02d\n", ETR_h, ETR_m, ETR_s) 

            # CFL
            u, v = compute_velocities(ψ)
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

"""
    h, m, s = hrs_mins_secs(seconds)

Returns hours `h`, minutes `m`, and seconds `s` equivalent to total number of `seconds`.
"""
function hrs_mins_secs(seconds)
    return seconds ÷ 3600, (seconds % 3600) ÷ 60, seconds % 60
end

function quick_plot(u; filename, vmax=0)
    g = u.g
    fig, ax = plt.subplots(1, figsize=(3.2, 4.5))
    if vmax == 0
        vmax = maximum(abs(u))
    end
    im = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=true)
    # im = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true)
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
