struct EvolutionComponents{SM, VSM}#, A4, A5}
    # depth-weighted mass matrix
    HM::SM # sparse matrix
    HHM::SM

    # "stiffness" matrices for vertical diffusion
    K_cols::VSM # vector of sparse matrices

    # # advection arrays
    # Ax1::A4 # 4D array
    # Ay1::A4
    # Ax2::A5 # 5D array
    # Ay2::A5
    # Ax_HHM_SD::A4 
    # Ay_HHM_SD::A4

    # advection on or off?
    advection::Bool
end

"""
    evolution = EvolutionComponents(geom::Geometry, forcing::Forcing, advection)
"""
function EvolutionComponents(params::Params, geom::Geometry, forcing::Forcing, advection)
    # unpack
    δ₀ = params.δ₀
    σ = geom.σ
    nσ = geom.nσ
    g1 = geom.g1
    g2 = geom.g2
    g_sfc2 = geom.g_sfc2
    H = geom.H
    nσ = geom.nσ
    κ = forcing.κ

    # vertical diffusion
    K_cols = [build_K_col(σ, κ[get_col_inds(i, nσ)]) for i ∈ 1:g_sfc2.np]

    # depth-weighted mass matrix
    HM = build_HM(g2, H, nσ)
    HHM = build_HHM(g2, H, nσ)

    # if advection
    #     h = cbrt.(6/π*g1.J.dets) # diameter of sphere with volume equal to wedge
    #     δ = δ₀*h
    #     Ax1, Ay1, Ax2, Ay2, Ax_HHM_SD, Ay_HHM_SD = build_advection_arrays(g1, g2, δ, H, nσ)
    # else
    #     Ax1 = Ay1 = Ax_HHM_SD = Ay_HHM_SD = zeros(1, 1, 1, 1)
    #     Ax2 = Ay2 = zeros(1, 1, 1, 1, 1)
    # end

    # return EvolutionComponents(HM, HHM, K_cols, Ax1, Ay1, Ax2, Ay2, Ax_HHM_SD, Ay_HHM_SD, advection)
    return EvolutionComponents(HM, HHM, K_cols, advection)
end

function advection_off(e::EvolutionComponents)
    # return EvolutionComponents(e.HM, e.HHM, e.K_cols, e.Ax1, e.Ay1, e.Ax2, e.Ay2, e.Ax_HHM_SD, e.Ay_HHM_SD, false)
    return EvolutionComponents(e.HM, e.HHM, e.K_cols, false)
end

"""
    K_col = build_K_col(σ, κ)

Compute finite difference matrix `K_col` for diffusion RHS of buoyancy equation.
"""
function build_K_col(σ, κ)
    nσ = length(σ)
    K = Tuple{Int64,Int64,Float64}[]
    for j=2:nσ-1
        fd_σ  = mkfdstencil(σ[j-1:j+1], σ[j], 1)
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)
        κ_σ = fd_σ'*κ[j-1:j+1]
        # ∂σ(κ ∂σ(b)) = κ ∂σσ(b) + ∂σ(κ) ∂σ(b)
        push!(K, (j, j-1, κ[j]*fd_σσ[1] + κ_σ*fd_σ[1]))
        push!(K, (j, j,   κ[j]*fd_σσ[2] + κ_σ*fd_σ[2]))
        push!(K, (j, j+1, κ[j]*fd_σσ[3] + κ_σ*fd_σ[3]))
    end
    # ∂σ(b) = 0 at z = -H
    fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
    push!(K, (1, 1, fd_σ[1]))
    push!(K, (1, 2, fd_σ[2]))
    push!(K, (1, 3, fd_σ[3]))
    # # ∂σ(b) = 0 at z = 0
    # fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    # push!(K, (nσ, nσ-2, fd_σ[1]))
    # push!(K, (nσ, nσ-1, fd_σ[2]))
    # push!(K, (nσ, nσ  , fd_σ[3]))
    # b = 0 at z = 0
    push!(K, (nσ, nσ, 1))
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nσ, nσ))
end

"""
    LHSs, RHSs = build_diffusion_matrices(m::ModelSetup3D, Δt)
"""
function build_diffusion_matrices(m::ModelSetup3D, Δt)
    # unpack
    ε² = m.params.ε²
    μϱ = m.params.μϱ
    nσ = m.geom.nσ
    H = m.geom.H
    g_sfc2 = m.geom.g_sfc2
    K_cols = m.evolution.K_cols

    α = ε²/μϱ
    LHSs_sp = [sparse(1.0*I(nσ)) for i=1:g_sfc2.np]
    RHSs = [sparse(zeros(nσ, nσ)) for i=1:g_sfc2.np]
    for i=1:g_sfc2.np
        if i ∉ g_sfc2.e["bdy"]
            LHSs_sp[i] = I - α/H[i]^2*Δt/4*K_cols[i] # Δt = Δt/2
            LHSs_sp[i][1, :] = K_cols[i][1, :]
            LHSs_sp[i][nσ, :] = K_cols[i][nσ, :]
            RHSs[i] = I + α/H[i]^2*Δt/4*K_cols[i]
            RHSs[i][1, :] .= 0
            RHSs[i][nσ, :] .= 0
        end
    end
    LHSs = [lu(LHS) for LHS ∈ LHSs_sp]

    return LHSs, RHSs
end

"""
    HM = build_HM(g2, H, nσ)

Compute `HM` = ∫ `H` φᵢ φⱼ for second order 3D grid `g2` with `nσ` vertical nodes.
"""
function build_HM(g2, H::FEField, nσ)
    # unpack
    g_sfc2 = H.g
    tri2 = g_sfc2.el
    w2 = g2.el

    # compute general integrals
    f(ξ, i, j, l) = φ(w2, ξ, i)*φ(w2, ξ, j)*φ(tri2, ξ[1:2], l)
    A = [ref_el_quad(ξ -> f(ξ, i, j, l), w2) for i ∈ 1:w2.n, j ∈ 1:w2.n, l ∈ 1:tri2.n]

    # stamp
    N = g2.nt*w2.n^2
    I = zeros(Int64, N)
    J = zeros(Int64, N)
    V = zeros(Float64, N)
    n = 1
    @showprogress "Building depth-weighted mass matrix..." for k=1:g2.nt, i=1:w2.n, j=1:w2.n
        I[n] = g2.t[k, i]
        J[n] = g2.t[k, j]
        for l ∈ 1:tri2.n
            V[n] += g2.J.dets[k]*H[g_sfc2.t[get_k_sfc(k, nσ), l]]*A[i, j, l]
        end
        n += 1
    end
    return dropzeros!(sparse(I, J, V, g2.np, g2.np))
end

"""
    HHM = build_HHM(g2, H, nσ)

Compute `HHM` = ∫ H² φᵢ φⱼ for second order 3D grid `g2` with `nσ` vertical nodes.
"""
function build_HHM(g2, H::FEField, nσ)
    # unpack
    g_sfc2 = H.g
    tri2 = g_sfc2.el
    w2 = g2.el

    # compute general integrals
    f(ξ, i, j, l, m) = φ(w2, ξ, i)*φ(w2, ξ, j)*φ(tri2, ξ[1:2], l)*φ(tri2, ξ[1:2], m)
    A = [ref_el_quad(ξ -> f(ξ, i, j, l, m), w2) for i ∈ 1:w2.n, j ∈ 1:w2.n, l ∈ 1:tri2.n, m ∈ 1:tri2.n]

    # stamp
    N = g2.nt*w2.n^2
    I = zeros(Int64, N)
    J = zeros(Int64, N)
    V = zeros(Float64, N)
    n = 1
    @showprogress "Building H²M..." for k=1:g2.nt, i=1:w2.n, j=1:w2.n
        I[n] = g2.t[k, i]
        J[n] = g2.t[k, j]
        for l ∈ 1:tri2.n, m ∈ 1:tri2.n
            V[n] += g2.J.dets[k]*H[g_sfc2.t[get_k_sfc(k, nσ), l]]*H[g_sfc2.t[get_k_sfc(k, nσ), m]]*A[i, j, l, m]
        end
        n += 1
    end
    return dropzeros!(sparse(I, J, V, g2.np, g2.np))
end

# function build_advection_arrays(g1, g2, δ, H, nσ)
#     # unpack
#     el1 = g1.el
#     el2 = g2.el
#     w = el1.quad_wts
#     qp = el1.quad_pts
#     φ1 = g1.φ_qp
#     φξ1 = g1.∂φ_qp[:, :, 1, :]
#     φη1 = g1.∂φ_qp[:, :, 2, :]
#     φσ1 = g1.∂φ_qp[:, :, 3, :]
#     φ2 = g2.φ_qp
#     φξ2 = g2.∂φ_qp[:, :, 1, :]
#     φη2 = g2.∂φ_qp[:, :, 2, :]
#     φσ2 = g2.∂φ_qp[:, :, 3, :]
#     Δ = g1.J.dets
#     nt = g1.nt

#     # allocate
#     println("I hope you have $((4*nt*el2.n*el1.n*el2.n + 2*nt*el2.n*el1.n*el1.n*el2.n)*8/1e9)G of memory...")
#     flush(stdout)

#     Ax1 = zeros(nt, el2.n, el1.n, el2.n)
#     Ay1 = zeros(nt, el2.n, el1.n, el2.n)
#     Ax2 = zeros(nt, el2.n, el1.n, el1.n, el2.n)
#     Ay2 = zeros(nt, el2.n, el1.n, el1.n, el2.n)
#     Ax_HHM_SD = zeros(nt, el2.n, el1.n, el2.n)
#     Ay_HHM_SD = zeros(nt, el2.n, el1.n, el2.n)

#     # uξ*∂ξ(b) + uη*∂η(b) + uσ*∂σ(b) = -∂σ(χy)*∂ξ(b) + ∂σ(χx)*∂η(b) + [∂ξ(χy) - ∂η(χx)]*∂σ(b) 
#     @showprogress "Building advection arrays..." for k ∈ 1:nt, i ∈ 1:el2.n, iχ1 ∈ 1:el1.n, ib ∈ 1:el2.n, i_quad ∈ eachindex(w)
#         k_sfc = get_k_sfc(k, nσ)

#         # ∂σ(χx)*∂η(b) - ∂η(χx)*∂σ(b)
#         Ax1[k, i, iχ1, ib]       +=      w[i_quad]*(φσ1[k, iχ1, i_quad]*φη2[k, ib, i_quad] - φη1[k, iχ1, i_quad]*φσ2[k, ib, i_quad])*φ2[i,  i_quad]*H(qp[i_quad, :], k_sfc)*Δ[k]
#         Ax_HHM_SD[k, i, iχ1, ib] += δ[k]*w[i_quad]*(φσ1[k, iχ1, i_quad]*φη2[k, i,  i_quad] - φη1[k, iχ1, i_quad]*φσ2[k, i,  i_quad])*φ2[ib, i_quad]*H(qp[i_quad, :], k_sfc)*Δ[k]

#         # ∂ξ(χy)*∂σ(b) - ∂σ(χy)*∂ξ(b)
#         Ay1[k, i, iχ1, ib]       +=      w[i_quad]*(φξ1[k, iχ1, i_quad]*φσ2[k, ib, i_quad] - φσ1[k, iχ1, i_quad]*φξ2[k, ib, i_quad])*φ2[i,  i_quad]*H(qp[i_quad, :], k_sfc)*Δ[k]
#         Ay_HHM_SD[k, i, iχ1, ib] += δ[k]*w[i_quad]*(φξ1[k, iχ1, i_quad]*φσ2[k, i,  i_quad] - φσ1[k, iχ1, i_quad]*φξ2[k, i,  i_quad])*φ2[ib, i_quad]*H(qp[i_quad, :], k_sfc)*Δ[k]

#         for iχ2 ∈ 1:el1.n
#             Ax2[k, i, iχ1, iχ2, ib] += δ[k]*w[i_quad]*(φσ1[k, iχ1, i_quad]*φη2[k, ib, i_quad] - φη1[k, iχ1, i_quad]*φσ2[k, ib, i_quad])*
#                                                       (φσ1[k, iχ2, i_quad]*φη2[k, i,  i_quad] - φη1[k, iχ2, i_quad]*φσ2[k, i,  i_quad])*Δ[k]
#             Ay2[k, i, iχ1, iχ2, ib] += δ[k]*w[i_quad]*(φξ1[k, iχ1, i_quad]*φσ2[k, ib, i_quad] - φσ1[k, iχ1, i_quad]*φξ2[k, ib, i_quad])*
#                                                       (φξ1[k, iχ2, i_quad]*φσ2[k, i,  i_quad] - φσ1[k, iχ2, i_quad]*φξ2[k, i,  i_quad])*Δ[k]
#         end
#     end

#     return CuArray(Ax1), CuArray(Ay1), CuArray(Ax2), CuArray(Ay2), CuArray(Ax_HHM_SD), CuArray(Ay_HHM_SD)
# end

#### 

# function build_HHM_SD(m::ModelSetup3D, HHM_SD_I, HHM_SD_J, b, χx, χy, t2)
#     # unpack
#     g2 = m.geom.g2
#     Ax_HHM_SD = m.evolution.Ax_HHM_SD
#     Ay_HHM_SD = m.evolution.Ay_HHM_SD

#     # load arrays on GPU
#     HHM_SD_V = CUDA.zeros(g2.nn, g2.nn, g2.nt)  # transpose in order to flatten in column-major order
#     χx_gpu = CuArray(χx)
#     χy_gpu = CuArray(χy)
#     b_gpu = CuArray(b)

#     # setup kernel
#     kernel = @cuda launch=false gpu_HHM_SD!(HHM_SD_V, Ax_HHM_SD, Ay_HHM_SD, b_gpu, χx_gpu, χy_gpu, t2)
#     config = launch_configuration(kernel.fun)
#     threads = min(g2.nt, config.threads)
#     blocks = cld(g2.nt, threads)

#     CUDA.@sync begin
#         kernel(HHM_SD_V, Ax_HHM_SD, Ay_HHM_SD, b_gpu, χx_gpu, χy_gpu, t2; threads, blocks)
#     end

#     # sparse matrix
#     return CuSparseMatrixCSC(HHM_SD_I, HHM_SD_J, HHM_SD_V[:], (g2.np, g2.np))
# end
# function build_HHM_SD_IJ(g2)
#     HHM_SD_I = zeros(Int32, g2.nt*g2.nn*g2.nn) 
#     HHM_SD_J = zeros(Int32, g2.nt*g2.nn*g2.nn) 
#     n = 1
#     for k ∈ 1:g2.nt, i ∈ 1:g2.nn, ib ∈ 1:g2.nn
#         HHM_SD_I[n] = g2.t[k, i]
#         HHM_SD_J[n] = g2.t[k, ib]
#         n += 1
#     end
#     HHM_SD_I = CuArray(HHM_SD_I)
#     HHM_SD_J = CuArray(HHM_SD_J)
#     return HHM_SD_I, HHM_SD_J
# end
# function gpu_HHM_SD!(HHM_SD_V, Ax_HHM_SD, Ay_HHM_SD, b, χx, χy, t2)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for k ∈ index:stride:size(Ax_HHM_SD, 1), i ∈ axes(Ax_HHM_SD, 2), iχ ∈ axes(Ax_HHM_SD, 3), ib ∈ axes(Ax_HHM_SD, 4)
#         HHM_SD_V[ib, i, k] += (Ax_HHM_SD[k, i, iχ, ib]*χx[k, iχ] + Ay_HHM_SD[k, i, iχ, ib]*χy[k, iχ])*b[t2[k, ib]] # transpose in order to flatten in column-major order
#     end
#     return 
# end

####

# function gpu_adv!(adv, Ax, Ay, χx, χy, b, t2)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for k ∈ index:stride:size(Ax, 1), i ∈ axes(Ax, 2), iχ ∈ axes(Ax, 3), ib ∈ axes(Ax, 4)
#         adv[k, i] += (Ax[k, i, iχ, ib]*χx[k, iχ] + Ay[k, i, iχ, ib]*χy[k, iχ])*b[t2[k, ib]]
#     end
#     return 
# end
# function gpu_adv!(adv, Ax1, Ay1, Ax2, Ay2, χx, χy, b, t2)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for k ∈ index:stride:size(Ax1, 1), i ∈ axes(Ax1, 2), iχ1 ∈ axes(Ax1, 3), ib ∈ axes(Ax1, 4)
#         adv[k, i] += (Ax1[k, i, iχ1, ib]*χx[k, iχ1] + Ay1[k, i, iχ1, ib]*χy[k, iχ1])*b[t2[k, ib]]
#         for iχ2 ∈ axes(Ax2, 4)
#             adv[k, i] += (Ax2[k, i, iχ1, iχ2, ib]*χx[k, iχ1]*χx[k, iχ2] + Ay2[k, i, iχ1, iχ2, ib]*χy[k, iχ1]*χy[k, iχ2])*b[t2[k, ib]]
#         end
#     end
#     return 
# end
# function advection(m::ModelSetup3D, χx, χy, b, t2)
#     # unpack
#     g2 = m.geom.g2
#     Ax1 = m.evolution.Ax1
#     Ay1 = m.evolution.Ay1
#     Ax2 = m.evolution.Ax2
#     Ay2 = m.evolution.Ay2

#     # load arrays on GPU
#     adv = CUDA.zeros(g2.nt, g2.nn) 
#     χx_gpu = CuArray(χx)
#     χy_gpu = CuArray(χy)
#     b_gpu = CuArray(b)

#     # setup advection kernel
#     kernel = @cuda launch=false gpu_adv!(adv, Ax1, Ay1, χx_gpu, χy_gpu, b_gpu, t2)
#     # kernel = @cuda launch=false gpu_adv!(adv, Ax1, Ay1, Ax2, Ay2, χx_gpu, χy_gpu, b_gpu, t2)
#     config = launch_configuration(kernel.fun)
#     threads = min(g2.nt, config.threads)
#     blocks = cld(g2.nt, threads)
#     # println(threads)
#     # println(blocks)

#     CUDA.@sync begin
#         kernel(adv, Ax1, Ay1, χx_gpu, χy_gpu, b_gpu, t2; threads, blocks)
#         # kernel(adv, Ax1, Ay1, Ax2, Ay2, χx_gpu, χy_gpu, b_gpu, t2; threads, blocks)
#         # @cuda threads=threads blocks=cld(g2.nt, threads) gpu_adv!(adv, Ax1, Ay1, χx_gpu, χy_gpu, b_gpu, t2)
#     end

#     # copy result to CPU
#     cpu_adv = Array(adv)
#     return cpu_adv
# end

# function advection(Ax1, Ay1, Ax2, Ay2, b, χx, χy)
#     # unpack
#     g1 = χx.g
#     g2 = b.g
#     el1 = g1.el
#     el2 = g2.el
#     nt = g1.nt

#     adv = zeros(g2.np)
#     for k ∈ 1:nt, i ∈ 1:el2.n, iχ1 ∈ 1:el1.n, ib ∈ 1:el2.n
#         adv[g2.t[k, i]] += (Ax1[k, i, iχ1, ib]*χx[k, iχ1] + Ay1[k, i, iχ1, ib]*χy[k, iχ1])*b[g2.t[k, ib]]
#         for iχ2 ∈ 1:el1.n
#             adv[g2.t[k, i]] += (Ax2[k, i, iχ1, iχ2, ib]*χx[k, iχ1]*χx[k, iχ2] + Ay2[k, i, iχ1, iχ2, ib]*χy[k, iχ1]*χy[k, iχ2])*b[g2.t[k, ib]]
#         end
#     end
#     return adv
# end

function gpu_adv!(adv, δ, w, φ2, φξ1, φη1, φσ1, φξ2, φη2, φσ2, H, Δ, χx, χy, b, t2, nσ)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for k ∈ index:stride:length(Δ), i ∈ axes(t2, 2), iχ1 ∈ axes(χx, 2), ib ∈ axes(t2, 2), i_quad ∈ eachindex(w)
        k_sfc = get_k_sfc(k, nσ)
        adv[k, i] += w[i_quad]*(φσ1[k, iχ1, i_quad]*φη2[k, ib, i_quad] - φη1[k, iχ1, i_quad]*φσ2[k, ib, i_quad])*φ2[i, i_quad]*H[k_sfc, i_quad]*Δ[k]*χx[k, iχ1]*b[t2[k, ib]]
        adv[k, i] += w[i_quad]*(φξ1[k, iχ1, i_quad]*φσ2[k, ib, i_quad] - φσ1[k, iχ1, i_quad]*φξ2[k, ib, i_quad])*φ2[i, i_quad]*H[k_sfc, i_quad]*Δ[k]*χy[k, iχ1]*b[t2[k, ib]]
        # for iχ2 ∈ axes(χx, 2)
        #     adv[k, i] += δ[k]*w[i_quad]*(φσ1[k, iχ1, i_quad]*φη2[k, ib, i_quad] - φη1[k, iχ1, i_quad]*φσ2[k, ib, i_quad])*
        #                                 (φσ1[k, iχ2, i_quad]*φη2[k, i,  i_quad] - φη1[k, iχ2, i_quad]*φσ2[k, i,  i_quad])*Δ[k]*χx[k, iχ1]*χx[k, iχ2]*b[t2[k, ib]]
        #     adv[k, i] += δ[k]*w[i_quad]*(φξ1[k, iχ1, i_quad]*φσ2[k, ib, i_quad] - φσ1[k, iχ1, i_quad]*φξ2[k, ib, i_quad])*
        #                                 (φξ1[k, iχ2, i_quad]*φσ2[k, i,  i_quad] - φσ1[k, iχ2, i_quad]*φξ2[k, i,  i_quad])*Δ[k]*χy[k, iχ1]*χy[k, iχ2]*b[t2[k, ib]]
        # end
    end
    return 
end
function advection(m::ModelSetup3D, δ, w, φ2, φξ1, φη1, φσ1, φξ2, φη2, φσ2, H, Δ, χx, χy, b, t2, nσ)
    # unpack
    g2 = m.geom.g2

    # load arrays on GPU
    adv = CUDA.zeros(g2.nt, g2.nn) 
    χx_gpu = CuArray(χx)
    χy_gpu = CuArray(χy)
    b_gpu = CuArray(b)

    # setup advection kernel
    kernel = @cuda launch=false gpu_adv!(adv, δ, w, φ2, φξ1, φη1, φσ1, φξ2, φη2, φσ2, H, Δ, χx_gpu, χy_gpu, b_gpu, t2, nσ)
    config = launch_configuration(kernel.fun)
    threads = min(g2.nt, config.threads)
    blocks = cld(g2.nt, threads)

    CUDA.@sync begin
        kernel(adv, δ, w, φ2, φξ1, φη1, φσ1, φξ2, φη2, φσ2, H, Δ, χx_gpu, χy_gpu, b_gpu, t2, nσ; threads, blocks)
    end

    # copy result to CPU
    cpu_adv = Array(adv)
    return cpu_adv
end

function evolve!(m::ModelSetup3D, s::ModelState3D, t_final, t_plot, t_save; Δt, i_save=0, i_plot=0)
    # unpack
    g1 = m.geom.g1
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    g_sfc2 = m.geom.g_sfc2
    HM = m.evolution.HM
    HHM = m.evolution.HHM
    advection_on = m.evolution.advection
    t2_gpu = CuArray(g2.t)
    if advection_on
        # δ = m.params.δ₀*cbrt.(6/π*g1.J.dets)
        # A, A_HM_SD = build_advection_arrays_fixed_flow(g2, δ, H, m.geom.Hx, nσ)
        # Pinv = sparse(inv(Diagonal(HM)))
        # adv = zeros(g2.np)
        # adv_prev = zeros(g2.np) 

        HHM_gpu = CuSparseMatrixCSC(HHM)
        # HHM_SD_I, HHM_SD_J = build_HHM_SD_IJ(g2)
        Pinv = CuSparseMatrixCSC(sparse(inv(Diagonal(HHM))))
        adv = CUDA.zeros(eltype(HHM_gpu), g2.np) # pre-allocate for `cg!`
        adv_prev = zeros(eltype(HHM_gpu), g2.np) 
        # CUDA.memory_status()

        # δ = CuArray(m.params.δ₀*cbrt.(6/π*g1.J.dets))
        δ = 1
        w = CuArray(g1.el.quad_wts)
        φξ1 = CuArray(g1.∂φ_qp[:, :, 1, :])
        φη1 = CuArray(g1.∂φ_qp[:, :, 2, :])
        φσ1 = CuArray(g1.∂φ_qp[:, :, 3, :])
        φ2 = CuArray(g2.φ_qp)
        φξ2 = CuArray(g2.∂φ_qp[:, :, 1, :])
        φη2 = CuArray(g2.∂φ_qp[:, :, 2, :])
        φσ2 = CuArray(g2.∂φ_qp[:, :, 3, :])
        Δ = CuArray(g1.J.dets)
        H_gpu = CuArray([H(g1.el.quad_pts[i_quad, :], k_sfc) for k_sfc ∈ 1:g_sfc2.nt, i_quad ∈ eachindex(w)])
        CUDA.memory_status()

        # Pinv = sparse(inv(Diagonal(HM)))
        # adv = zeros(g2.np) # pre-allocate for `cg!`
        # adv_prev = zeros(g2.np) 
    end

    # stiffness matrix for vertical diffusion
    LHS_diffs, RHS_diffs = build_diffusion_matrices(m, Δt)

    # element map TODO: add this to `Grid`?
    el_map = build_element_map(g2)

    # timestep
    t_current = s.t[1]
    n_steps = Int64(round((t_final - t_current)/Δt))
    n_steps_plot = Int64(round(t_plot/Δt))
    n_steps_save = Int64(round(t_save/Δt))

    # # fixed flow: u = 1, v = w = 0 → χx = 0, χy = -σH
    # s.χx.values[:] .= 0
    # s.χy.values[:] = -g1.p[g1.t, 3].*H[get_i_sfc.(g1.t, nσ)]
    # plot_u(m, s, 0)

    # initial condition
    ∫b₀ = sum(HM*s.b.values)
    pe₀ = potential_energy(m, s)
    println("\nInitial condition:") 
    @printf("    ∫b₀ = % .5e\n    pe₀ = % .5e\n", ∫b₀, pe₀)
    quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", title=latexstring(L"$t = $", @sprintf("%1.1e", s.t[1])), filename=@sprintf("%s/images/psi%03d.png", out_folder, i_plot))
    plot_u(m, s, 0; i=i_plot)
    plot_profiles(m, s, x=0.5, y=0.0, filename=@sprintf("%s/images/profiles%03d.png", out_folder, i_plot))
    i_plot += 1
    save_state(s, "$out_folder/data/state$i_save.h5")
    i_save += 1

    # for CFL
    dx = [sum(abs(g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), mod1(i+1, 3)], 1] - g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), i], 1]) for i=1:3)/3 for k=1:g1.nt]
    dy = [sum(abs(g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), mod1(i+1, 3)], 2] - g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), i], 2]) for i=1:3)/3 for k=1:g1.nt]
    dσ = [abs(g1.p[g1.t[k, 1], 3] - g1.p[g1.t[k, 4], 3]) for k=1:g1.nt]
    ux = [-∂σ(s.χy, [0.25, 0.25, 0.5], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
    uy = [+∂σ(s.χx, [0.25, 0.25, 0.5], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
    uσ = [(∂ξ(s.χy, [0.25, 0.25, 0.5], k) - ∂η(s.χx, [0.25, 0.25, 0.5], k))/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
    println("CFL:") 
    @printf("    Δt_x = %.5e\n    Δt_y = %.5e\n    Δt_σ = %.5e\n    Δt   = %.5e\n", minimum(dx./abs.(ux)), minimum(dy./abs.(uy)), minimum(dσ./abs.(uσ)), Δt)
    flush(stdout)
    flush(stderr)

    # solve
    t0 = time()
    for i ∈ 1:n_steps
        # Δt/2 vertical diffusion step
        # @time "vdiff" begin
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end
        # end

        # Δt advection step
        # @time "adv" begin
        if advection_on
            # invert
            invert!(m, s)
            
            # @time "\tadv" begin
            # adv_el = advection(m, s.χx.values, s.χy.values, s.b.values, t2_gpu)
            adv_el = advection(m, δ, w, φ2, φξ1, φη1, φσ1, φξ2, φη2, φσ2, H_gpu, Δ, s.χx.values, s.χy.values, s.b.values, t2_gpu, nσ)
            adv_node_gpu = CuArray(el_map*adv_el[:])
            # end
            # HHM_SD = build_HHM_SD(m, HHM_SD_I, HHM_SD_J, s.b.values, s.χx.values, s.χy.values, t2_gpu)
            # cg!(adv, HHM_gpu + HHM_SD, -adv_node_gpu; Pinv)
            cg!(adv, HHM_gpu, -adv_node_gpu; Pinv)

            if i == 1
                # euler first step
                s.b.values[:] = s.b.values + Δt*Array(adv)
            else
                # AB2 otherwise
                s.b.values[:] = s.b.values + 3/2*Δt*Array(adv) - 1/2*Δt*adv_prev
            end

            # # for advection only: set b = 0 on coast
            # s.b.values[g2.e["coast"]] .= 0

            # save for AB2
            adv_prev[:] = Array(adv)[:]
        end
        # end

        # Δt/2 diffusion step
        # @time "vdiff" begin
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end
        # end

        # set time
        s.t[1] = s.t[1] + Δt

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end

        if mod(i, n_steps_plot) == 0 || i == n_steps
            # timing
            elapsed = time() - t0
            elapsed_h, elapsed_m, elapsed_s = hrs_mins_secs(elapsed)
            ETR = (n_steps - i)*elapsed/i
            ETR_h, ETR_m, ETR_s = hrs_mins_secs(ETR)
            @printf("\nt = %.2e (%d/%d steps)\n", s.t[1], i, n_steps)
            @printf("    time elapsed:   %02d:%02d:%02d\n", elapsed_h, elapsed_m, elapsed_s)
            @printf("    time remaining: %02d:%02d:%02d\n", ETR_h, ETR_m, ETR_s) 

            # buoyancy
            ∫b = sum(HM*s.b.values) 
            Δb = abs(∫b - ∫b₀) 
            Δb_pct = 100*abs(Δb/∫b₀)
            println("Buoyancy conservation:") 
            @printf("    ∫b     = % .5e\n    Δb     = % .5e\n    Δb_pct = % .5e\n", ∫b, Δb, Δb_pct)
            if Δb_pct > 100
                error("Solution blew up 😢")
            end

            # CFL
            ux = [-∂σ(s.χy, [0.25, 0.25, 0.5], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            uy = [+∂σ(s.χx, [0.25, 0.25, 0.5], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            uσ = [(∂ξ(s.χy, [0.25, 0.25, 0.5], k) - ∂η(s.χx, [0.25, 0.25, 0.5], k))/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            println("CFL:") 
            @printf("    Δt_x = %.5e\n    Δt_y = %.5e\n    Δt_σ = %.5e\n    Δt   = %.5e\n", minimum(dx./abs.(ux)), minimum(dy./abs.(uy)), minimum(dσ./abs.(uσ)), Δt)

            # # energy 
            # b_prod = buoyancy_production(m, s) 
            # ke_diss = KE_dissipation(m, s)
            # println("KE:")
            # @printf("    ∫uᶻb   = % .5e\n", b_prod)
            # @printf("    ε²∫νω² = % .5e\n", ke_diss)
            # @printf("    error  = % .5e\n", abs(ke_diss - b_prod))
            # pe = potential_energy(m, s)
            # pe_prod = PE_production(m, s)
            # println("PE:")
            # @printf("    pe      = % .5e\n", pe)
            # @printf("    Δpe     = % .5e\n", pe - pe₀)
            # @printf("    pe_prod = % .5e\n", pe_prod)

            # # advection solution
            # ba = FEField([ba_adv(g2.p[j, :], i*Δt, H[get_i_sfc(j, nσ)]) for j=1:g2.np], g2)
            # @printf "Max Err = %1.1e\n" maximum(abs(s.b - ba))
            # # diffusion solution
            # ba = [ba_diff(g2.p[j, 3], i*Δt, α/(1-g2.p[j, 1]^2-g2.p[j, 2]^2)^2, 1-g2.p[j, 1]^2-g2.p[j, 2]^2) for j=1:g2.np]

            # debug plot
            if !m.evolution.advection
                invert!(m, s)
            end
            quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", title=latexstring(L"$t = $", @sprintf("%1.1e", s.t[1])), filename=@sprintf("%s/images/psi%03d.png", out_folder, i_plot))
            plot_u(m, s, 0; i=i_plot)
            plot_profiles(m, s, x=0.5, y=0.0, filename=@sprintf("%s/images/profiles%03d.png", out_folder, i_plot))
            # plot_xslice(m, s.b, abs(s.b - ba), 0, L"|b - b_a|", @sprintf("%s/images/b%03d.png", out_folder, i_plot))
            # plot_xslice(m, s.b, s.b, 0.0, L"b", @sprintf("%s/images/b%03d.png", out_folder, i_plot))
            i_plot += 1

            flush(stdout)
            flush(stderr)
        end
        if mod(i, n_steps_save) == 0 || i == n_steps
            save_state(s, "$out_folder/data/state$i_save.h5")
            i_save += 1
            flush(stdout)
            flush(stderr)
        end

    end

    return s
end

"""
    h, m, s = hrs_mins_secs(seconds)

Returns hours `h`, minutes `m`, and seconds `s` equivalent to total number of `seconds`.
"""
function hrs_mins_secs(seconds)
    return seconds ÷ 3600, (seconds % 3600) ÷ 60, seconds % 60
end

"""
    ba = ba_diff(σ, t, α, H; N)

Analytical solution to ∂t(b) = α ∂σσ(b) with ∂σ(b) = 0 at σ = -1, 0
and b(σ, 0) = H*σ (truncated to Nth term in Fourier series).
"""
function ba_diff(σ, t, α, H; N=1000)
    # b0 = H*σ
    A(n) = 2*H*(1 + (-1)^(n+1))/(n^2*π^2)
    return -H/2 + sum(A(n)*cos(n*π*σ)*exp(-α*(n*π)^2*t) for n=1:2:N)
end

"""
    ba = ba_adv(x, t)

Analytical solution to ∂t(b) + ∂x(b) = 0 for gaussian initial condition.
"""
function ba_adv(x, t, H)
    return -exp(-((x[1] - t)^2 + x[2]^2 + (H*x[3] + 0.5)^2)/0.02)
end

# function build_advection_arrays_fixed_flow(g2, δ, H, Hx, nσ)
#     # unpack
#     el2 = g2.el
#     w = el2.quad_wts
#     qp = el2.quad_pts
#     φ2 = g2.φ_qp
#     φξ2 = g2.∂φ_qp[:, :, 1, :]
#     φσ2 = g2.∂φ_qp[:, :, 3, :]
#     Δ = g2.J.dets
#     nt = g2.nt

#     # fixed flow: u = 1, v = w = 0 → χx = 0, χy = -σH
#     A = zeros(nt, el2.n, el2.n)
#     A_HM_SD = zeros(nt, el2.n, el2.n)
#     @showprogress "Building advection arrays..." for k ∈ 1:nt, i ∈ 1:el2.n, ib ∈ 1:el2.n, i_quad ∈ eachindex(w)
#         # -σHₓ*∂σ(b) + H*∂ξ(b)
#         k_sfc = get_k_sfc(k, nσ)
#         A[k, i, ib]       +=      w[i_quad]*(H(qp[i_quad, :], k_sfc)*φξ2[k, ib, i_quad] - qp[i_quad, 3]*Hx(qp[i_quad, :], k_sfc)*φσ2[k, ib, i_quad])*φ2[i,  i_quad]*Δ[k] +
#                              δ[k]*w[i_quad]*(H(qp[i_quad, :], k_sfc)*φξ2[k, ib, i_quad] - qp[i_quad, 3]*Hx(qp[i_quad, :], k_sfc)*φσ2[k, ib, i_quad])*
#                                             (H(qp[i_quad, :], k_sfc)*φξ2[k, i,  i_quad] - qp[i_quad, 3]*Hx(qp[i_quad, :], k_sfc)*φσ2[k, i,  i_quad])*Δ[k]/H(qp[i_quad, :], k_sfc)
#         A_HM_SD[k, i, ib] += δ[k]*w[i_quad]*(H(qp[i_quad, :], k_sfc)*φξ2[k, i,  i_quad] - qp[i_quad, 3]*Hx(qp[i_quad, :], k_sfc)*φσ2[k, i,  i_quad])*φ2[ib, i_quad]*Δ[k]
#     end
#     return A, A_HM_SD
# end
# function advection_fixed_flow(A, b)
#     # unpack
#     g2 = b.g
#     el2 = g2.el
#     nt = g2.nt

#     adv = zeros(g2.np)
#     for k ∈ 1:nt, i ∈ 1:el2.n, ib ∈ 1:el2.n
#         adv[g2.t[k, i]] += A[k, i, ib]*b[g2.t[k, ib]]
#     end
#     return adv
# end
# function build_HM_SD_fixed_flow(A_HM_SD, b)
#     # unpack
#     g2 = b.g
#     el2 = g2.el
#     nt = g2.nt

#     HM_SD_I = zeros(Int64, nt*el2.n*el2.n) 
#     HM_SD_J = zeros(Int64, nt*el2.n*el2.n) 
#     HM_SD_V = zeros(Float64, nt*el2.n*el2.n) 
#     n = 1
#     for k ∈ 1:nt, i ∈ 1:el2.n, ib ∈ 1:el2.n
#         HM_SD_I[n] = g2.t[k, i]
#         HM_SD_J[n] = g2.t[k, ib]
#         HM_SD_V[n] = A_HM_SD[k, i, ib]*b[g2.t[k, ib]]
#         n += 1
#     end
#     return dropzeros!(sparse(HM_SD_I, HM_SD_J, HM_SD_V, g2.np, g2.np))
# end