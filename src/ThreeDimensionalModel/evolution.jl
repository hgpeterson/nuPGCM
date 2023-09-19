"""
    K_col = get_K_col(σ, κ)

Compute finite difference matrix `K_col` for diffusion RHS of buoyancy equation.
"""
function get_K_col(σ, κ)
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
    # ∂σ(b) = 0 at z = 0
    fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    push!(K, (nσ, nσ-2, fd_σ[1]))
    push!(K, (nσ, nσ-1, fd_σ[2]))
    push!(K, (nσ, nσ  , fd_σ[3]))
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nσ, nσ))
end

"""
    HM = get_HM(g2, H, nσ)

Compute `HM` = ∫ `H` φᵢ φⱼ for second order 3D grid `g2` with `nσ` vertical nodes.
"""
function get_HM(g2, H::FEField, nσ)
    # unpack
    g_sfc2 = H.g
    tri2 = g_sfc2.el
    w2 = g2.el

    # compute general integrals
    f(ξ, i, j, k) = φ(w2, ξ, i)*φ(w2, ξ, j)*φ(tri2, ξ[1:2], k)
    A = [ref_el_quad(ξ -> f(ξ, i, j, k), w2) for i=1:w2.n, j=1:w2.n, k=1:tri2.n]

    # stamp
    N = g2.nt*w2.n^2
    I = zeros(Int64, N)
    J = zeros(Int64, N)
    V = zeros(Float64, N)
    n = 1
    @showprogress "Building depth-weighted mass matrix..." for k=1:g2.nt, i=1:w2.n, j=1:w2.n
        I[n] = g2.t[k, i]
        J[n] = g2.t[k, j]
        V[n] = g2.J.dets[k]*sum(A[i, j, :].*H[g_sfc2.t[get_k_sfc(k, nσ), :]])
        n += 1
    end
    return dropzeros!(sparse(I, J, V, g2.np, g2.np))
    # return Tridiagonal(dropzeros!(sparse(I, J, V, g2.np, g2.np)))
end

"""
    Ax, Ay = get_advection_arrays(g1, g2)

Compute advection arrays of the form ∫ φᵢ∂φⱼ∂φₖ where φᵢ and φⱼ are defined on the 
second order grid `g2` and φₖ is defined on the first order grid `g1`. These are then
multiplied by the proper Jacobian terms to get the arrays:
    • `Ax` for the  ∂σ(χx)*∂η(b) and -∂η(χx)*∂σ(b) terms,
    • `Ay` for the -∂σ(χy)*∂ξ(b) and  ∂ξ(χy)*∂σ(b) terms.
"""
function get_advection_arrays(g1, g2)
    # unpack
    J = g1.J
    w1 = g1.el
    w2 = g2.el

    # compute general integrals
    f(ξ, i, j, k, d1, d2) = ∂φ(w1, ξ, k, d1)*∂φ(w2, ξ, j, d2)*φ(w2, ξ, i)
    A = [ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), w1) for i=1:w2.n, j=1:w2.n, k=1:w1.n, d1=1:3, d2=1:3]

    # allocate
    Ax  = zeros(Float32, g1.nt, w2.n, w2.n, w1.n)
    Ay  = zeros(Float32, g1.nt, w2.n, w2.n, w1.n)

    @showprogress "Setting up advection arrays..." for k=1:g1.nt
        # unpack
        jac = J.Js[k, :, :]
        Δ = J.dets[k]

        for d1=1:3, d2=1:3
            # ∂σ(χx)*∂η(b) - ∂η(χx)*∂σ(b)
            Ax[k, :, :, :] += A[:, :, :, d1, d2]*(jac[d1, 3]*jac[d2, 2] - jac[d1, 2]*jac[d2, 3])*Δ

            # ∂ξ(χy)*∂σ(b) - ∂σ(χy)*∂ξ(b)
            Ay[k, :, :, :] += A[:, :, :, d1, d2]*(jac[d1, 1]*jac[d2, 3] - jac[d1, 3]*jac[d2, 1])*Δ
        end
    end

    Ax_gpu = CuArray(Ax)
    Ay_gpu = CuArray(Ay)

    return Ax_gpu, Ay_gpu
    # return Ax, Ay
end

# function advection(m::ModelSetup3D, χx, χy, b)
#     g1 = m.g1
#     g2 = m.g2
#     adv = zeros(g2.np)
#     for k=1:g2.nt, i=1:g2.nn
#         adv[g2.t[k, i]] += sum(m.Aξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χy[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
#                            sum(m.Aη[k, i, ib, iχ]*b[g2.t[k, ib]]*χx[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
#                            sum(m.Aσξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χy[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
#                            sum(m.Aση[k, i, ib, iχ]*b[g2.t[k, ib]]*χx[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn)
#     end
#     return adv
# end

# function advection_chunk!(adv, k_chunk, m::ModelSetup3D, χx, χy, b)
#     g1 = m.g1
#     g2 = m.g2
#     for k=k_chunk, i=1:g2.nn
#         adv[k, i] = sum(((m.Aξ[k, i, ib, iχ] + m.Aσξ[k, i, ib, iχ])*χy[k, iχ] +
#                          (m.Aη[k, i, ib, iχ] + m.Aση[k, i, ib, iχ])*χx[k, iχ])*b[g2.t[k, ib]] for ib=1:g2.nn, iχ=1:g1.nn)
#     end                                                
#     return adv
# end
# function advection(m::ModelSetup3D, χx, χy, b)
#     k_chunks = Iterators.partition(1:m.g1.nt, m.g1.nt ÷ Threads.nthreads())
#     adv = zeros(m.g1.nt, m.g2.nn)
#     tasks = map(k_chunks) do k_chunk
#         Threads.@spawn advection_chunk!(adv, k_chunk, m, χx, χy, b)
#     end
#     fetch.(tasks)
#     return [sum(adv[I] for I ∈ m.g2.p_to_t[i]) for i=1:m.g2.np] 
# end

function gpu_adv!(adv, Ax, Ay, χx, χy, b, t2)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    Is = CartesianIndices((axes(Ax, 1), axes(Ax, 2)))
    for i=index:stride:length(Is), ib ∈ axes(Ax, 3), iχ ∈ axes(Ax, 4)
        adv[t2[Is[i]]] += (Ax[Is[i], ib, iχ]*χx[Is[i][1], iχ] + Ay[Is[i], ib, iχ]*χy[Is[i][1], iχ])*b[t2[Is[i][1], ib]]
    end
    return 
end
function advection(m::ModelSetup3D, χx, χy, b)
    # load arrays on GPU
    adv = CUDA.zeros(m.g2.np) 
    χx_gpu = CuArray(χx.values)
    χy_gpu = CuArray(χy.values)
    b_gpu = CuArray(b.values)
    t2 = CuArray(m.g2.t)

    # setup advection kernel
    kernel = @cuda launch=false gpu_adv!(adv, m.Ax, m.Ay, χx_gpu, χy_gpu, b_gpu, t2)
    config = launch_configuration(kernel.fun)
    N = m.g2.nt*m.g2.nn
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    CUDA.@sync begin
        kernel(adv, m.Ax, m.Ay, χx_gpu, χy_gpu, b_gpu, t2; threads, blocks)
    end

    # copy result to CPU
    return Array(adv)
end

function evolve!(m::ModelSetup3D, s::ModelState3D, t_final, t_plot)
    # unpack
    μ = m.μ
    ϱ = m.ϱ
    ε² = m.ε²
    κ = m.κ
    Δt = m.Δt
    g1 = m.g1
    g2 = m.g2
    nσ = m.nσ
    H = m.H
    HM = m.HM
    g_sfc2 = m.g_sfc2

    # mass matrix for computing ∫b
    M = mass_matrix(g2)

    # stiffness matrix for stabilizing diffusion
    K_stab = stiffness_matrix(g2)
    LHS_stab = lu(I + 2e1*Δt*K_stab)

    # timestep
    Δt = 0.04
    n_steps = 1
    n_steps_plot = 1
    # n_steps = Int64(round(t_final/Δt))
    # n_steps_plot = Int64(round(t_plot/Δt))

    # diffusion matrices
    # α = ε²/μ/ϱ
    α = 1e-3
    K_cols = [get_K_col(m.σ, κ[get_col_inds(i, nσ)]) for i=1:g_sfc2.np]
    LHS_diffs_sp = [sparse(1.0*I(nσ)) for i=1:g_sfc2.np]
    RHS_diffs = [sparse(zeros(nσ, nσ)) for i=1:g_sfc2.np]
    for i=1:g_sfc2.np
        if i ∉ g_sfc2.e["bdy"]
            LHS_diffs_sp[i] = I - α/H[i]^2*Δt/4*K_cols[i] # Δt = Δt/2
            LHS_diffs_sp[i][1, :] = K_cols[i][1, :]
            LHS_diffs_sp[i][nσ, :] = K_cols[i][nσ, :]
            RHS_diffs[i] = I + α/H[i]^2*Δt/4*K_cols[i]
            RHS_diffs[i][1, :] .= 0
            RHS_diffs[i][nσ, :] .= 0
        end
    end
    LHS_diffs = [lu(LHS) for LHS ∈ LHS_diffs_sp]

    # pvd file
    rm("$out_folder/state.pvd", force=true)
    cmd = "rm -f $out_folder/state*.vtu"
    run(`bash -c $cmd`)
    pvd = paraview_collection("$out_folder/state", append=true)

    # for plotting
    pz = copy(g1.p)
    for i=1:g1.np
        pz[i, 3] *= H[get_i_sfc(i, nσ)]
    end

    # initial condition
    ∫b₀ = sum(M*s.b.values)
    @info "Initial condition" ∫b₀
    cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
    vtk_grid("$out_folder/state0", pz', cells) do vtk
        vtk["b"] = s.b.values[1:g1.np]
        vtk["ba"] = s.b.values[1:g1.np]
        vtk["err"] = zeros(g1.np)
        vtk["ωx"] = FEField(s.ωx).values
        vtk["ωy"] = FEField(s.ωy).values
        vtk["χx"] = FEField(s.χx).values
        vtk["χy"] = FEField(s.χy).values
        pvd[0] = vtk
    end
    println("$out_folder/state0.vtu")

    # for CFL
    dx = [sum(abs(g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), mod1(i+1, 3)], 1] - g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), i], 1]) for i=1:3)/3 for k=1:g1.nt]
    dy = [sum(abs(g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), mod1(i+1, 3)], 2] - g_sfc2.p[g_sfc2.t[get_k_sfc(k, nσ), i], 2]) for i=1:3)/3 for k=1:g1.nt]
    dσ = [abs(g1.p[g1.t[k, 1], 3] - g1.p[g1.t[k, 4], 3]) for k=1:g1.nt]
    ux = [-∂z(s.χy, [0, 0, 0], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
    uy = [+∂z(s.χx, [0, 0, 0], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
    uσ = [(∂x(s.χy, [0, 0, 0], k) - ∂y(s.χx, [0, 0, 0], k))/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
    @info "CFL" Δt_x=minimum(dx./abs.(ux)) Δt_y=minimum(dy./abs.(uy)) Δt_σ=minimum(dσ./abs.(uσ)) Δt

    # solve
    adv = zeros(g2.np) # pre-allocate for `cg!`
    t0 = time()
    for i=1:n_steps
        # println("step $i")

        # Δt/2 diffusion step
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end

        # Δt advection step
        if m.advection
            # invert!(m, s)
            @time cg!(adv, HM, -advection(m, s.χx, s.χy, s.b))
            # adv[:] = -(HM\advection(m, s.χx, s.χy, s.b))
            s.b.values[:] = s.b.values + Δt/2*adv
            # invert!(m, s)
            cg!(adv, HM, -advection(m, s.χx, s.χy, s.b))
            # adv[:] = -(HM\advection(m, s.χx, s.χy, s.b))
            s.b.values[:] = s.b.values + Δt*adv
        end

        # stabilizing diffusion
        s.b.values[:] = LHS_stab\s.b.values

        # Δt/2 diffusion step
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end

        if mod(i, n_steps_plot) == 0 || i == n_steps
            # # advection solution
            # ba = [ba_adv(g2.p[j, :], i*Δt, H[get_i_sfc(j, nσ)]) for j=1:g2.np]
            # # diffusion solution
            # ba = [ba_diff(g2.p[j, 3], i*Δt, α/(1-g2.p[j, 1]^2-g2.p[j, 2]^2)^2, 1-g2.p[j, 1]^2-g2.p[j, 2]^2) for j=1:g2.np]

            # info
            t_elapsed = time() - t0
            ∫b = sum(M*s.b.values) 
            Δb = abs(∫b - ∫b₀) 
            Δb_pct = 100*abs(Δb/∫b₀)
            @info @sprintf("%d steps in %d s (ETR: %.1f m)", i, t_elapsed, (n_steps-i)*t_elapsed/i/60) ∫b Δb Δb_pct
            # ux = [-∂z(s.χy, [0, 0, 0], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            # uy = [+∂z(s.χx, [0, 0, 0], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            # uσ = [(∂x(s.χy, [0, 0, 0], k) - ∂y(s.χx, [0, 0, 0], k))/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            # @info "CFL" Δt_x=minimum(dx./abs.(ux)) Δt_y=minimum(dy./abs.(uy)) Δt_σ=minimum(dσ./abs.(uσ)) Δt

            # show state
            # invert!(m, s, showplots=true)

            # save state
            cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
            vtk_grid("$out_folder/state$i", pz', cells) do vtk
                vtk["b"] = s.b.values[1:g1.np]
                # vtk["ba"] = ba[1:g1.np]
                # vtk["err"] = abs.(s.b.values[1:g1.np] - ba[1:g1.np])
                vtk["ωx"] = FEField(s.ωx).values
                vtk["ωy"] = FEField(s.ωy).values
                vtk["χx"] = FEField(s.χx).values
                vtk["χy"] = FEField(s.χy).values
                pvd[i*Δt] = vtk
            end
            println("$out_folder/state$i.vtu")
            cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
            vtk_grid("$out_folder/stateTF$i", g1.p', cells) do vtk
                vtk["b"] = s.b.values[1:g1.np]
            end
            println("$out_folder/stateTF$i.vtu")
        end
    end

    vtk_save(pvd)
    println("$out_folder/state.pvd")

    return s
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
    return exp(-((x[1] - t)^2 + x[2]^2 + (H*x[3] + 0.5)^2)/0.02)
end