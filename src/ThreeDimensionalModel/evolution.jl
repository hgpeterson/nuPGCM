struct EvolutionComponents{SM, VSM, A}
    # depth-weighted mass matrix
    HM::SM # sparse matrix

    # stiffness matrix for horizontal diffusion
    K_hdiff::SM

    # "stiffness" matrices for vertical diffusion
    K_cols::VSM # vector of sparse matrices

    # advection arrays
    Ax::A # 4D array
    Ay::A

    # advection on or off?
    advection::Bool
end

"""
    evolution = EvolutionComponents(geom::Geometry, forcing::Forcing, advection)
"""
function EvolutionComponents(geom::Geometry, forcing::Forcing, advection)
    # unpack
    σ = geom.σ
    nσ = geom.nσ
    g1 = geom.g1
    g2 = geom.g2
    g_sfc2 = geom.g_sfc2
    H = geom.H
    nσ = geom.nσ
    κ = forcing.κ

    # horizontal diffusion
    K_hdiff = build_K_hdiff(geom)
    # K_hdiff = spzeros(g2.np, g2.np)

    # vertical diffusion
    K_cols = [build_K_col(σ, κ[get_col_inds(i, nσ)]) for i ∈ 1:g_sfc2.np]

    # depth-weighted mass matrix
    HM = build_HM(g2, H, nσ)

    if advection
        Ax, Ay = build_advection_arrays(g1, g2)
    else
        Ax = Ay = zeros(1, 1, 1, 1)
    end

    return EvolutionComponents(HM, K_hdiff, K_cols, Ax, Ay, advection)
end

function advection_off(e::EvolutionComponents)
    return EvolutionComponents(e.HM, e.K_hdiff, e.K_cols, e.Ax, e.Ay, false)
end

function build_K_hdiff(geom::Geometry)
    # unpack
    g = geom.g2
    el = g.el
    nσ = geom.nσ
    H = geom.H
    Hx = geom.Hx
    Hy = geom.Hy

    # σ FE
    σ = FEField(g.p[:, 3], g)

    # integration function
    ∇φ_refs = [∂φ(el, el.quad_pts[i_quad, :], i, d) for d=1:3, i_quad ∈ eachindex(el.quad_wts), i=1:g.nn]
    function ∫f(i, j, k, jacs, Hs, Hxs, Hys, σs)
        ∇φi = jacs'*∇φ_refs[:, :, i]
        ∇φj = jacs'*∇φ_refs[:, :, j]
        fi = @. g.J.dets[k]*((∇φi[1, :]*Hs - σs*Hxs*∇φi[3, :])*(∇φj[1, :]*Hs - σs*Hxs*∇φj[3, :]) + 
                             (∇φi[2, :]*Hs - σs*Hys*∇φi[3, :])*(∇φj[2, :]*Hs - σs*Hys*∇φj[3, :]) + 
                              ∇φi[3, :]*∇φj[3, :]/Hs)
        return dot(el.quad_wts, fi)
    end

    # stamp
    N = g.nt*el.n^2
    I = zeros(Int64, N)
    J = zeros(Int64, N)
    V = zeros(Float64, N)
    n = 1
    @showprogress "Building horizontal diffusion matrix..." for k=1:g.nt
        jacs = g.J.Js[k, :, :]
        k_sfc = get_k_sfc(k, nσ)
        Hs = [H(el.quad_pts[i, 1:2], k_sfc) for i ∈ eachindex(el.quad_wts)]
        Hxs = [Hx(el.quad_pts[i, 1:2], k_sfc) for i ∈ eachindex(el.quad_wts)]
        Hys = [Hy(el.quad_pts[i, 1:2], k_sfc) for i ∈ eachindex(el.quad_wts)]
        σs = [σ(el.quad_pts[i, :], k) for i ∈ eachindex(el.quad_wts)]
        for i=1:el.n, j=1:el.n
            I[n] = g.t[k, i]
            J[n] = g.t[k, j]
            V[n] = ∫f(i, j, k, jacs, Hs, Hxs, Hys, σs)
            n += 1
        end
    end

    return dropzeros!(sparse(I, J, V, g.np, g.np))
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
end

"""
    Ax, Ay = build_advection_arrays(g1, g2)

Compute advection arrays of the form ∫ φᵢ∂φⱼ∂φₖ where φᵢ and φⱼ are defined on the 
second order grid `g2` and φₖ is defined on the first order grid `g1`. These are then
multiplied by the proper Jacobian terms to get the arrays:
    • `Ax` for the  ∂σ(χx)*∂η(b) and -∂η(χx)*∂σ(b) terms,
    • `Ay` for the -∂σ(χy)*∂ξ(b) and  ∂ξ(χy)*∂σ(b) terms.
"""
function build_advection_arrays(g1, g2)
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

    @showprogress "Building advection arrays..." for k=1:g1.nt
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
end

function gpu_adv!(adv, Ax, Ay, χx, χy, b, t2)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    Is = CartesianIndices((axes(Ax, 1), axes(Ax, 2)))
    for i = index:stride:length(Is)
        for ib ∈ axes(Ax, 3), iχ ∈ axes(Ax, 4)
            adv[Is[i]] += (Ax[Is[i], ib, iχ]*χx[Is[i][1], iχ] + Ay[Is[i], ib, iχ]*χy[Is[i][1], iχ])*b[t2[Is[i][1], ib]]
        end
    end
    return 
end
function advection(m::ModelSetup3D, χx, χy, b)
    # unpack
    g2 = m.geom.g2
    Ax = m.evolution.Ax
    Ay = m.evolution.Ay

    # load arrays on GPU
    adv = CUDA.zeros(g2.nt, g2.nn) 
    χx_gpu = CuArray(χx)
    χy_gpu = CuArray(χy)
    b_gpu = CuArray(b)
    t2 = CuArray(g2.t)

    # # setup advection kernel
    # kernel = @cuda launch=false gpu_adv!(adv, Ax, Ay, χx_gpu, χy_gpu, b_gpu, t2)
    # config = launch_configuration(kernel.fun)
    # N = g2.nt*g2.nn
    # threads = min(N, config.threads)
    # blocks = cld(N, threads)
    # println(threads)
    # println(blocks)

    CUDA.@sync begin
        # kernel(adv, m.Ax, m.Ay, χx_gpu, χy_gpu, b_gpu, t2; threads, blocks)
        @cuda threads=512 blocks=cld(length(t2), 512) gpu_adv!(adv, Ax, Ay, χx_gpu, χy_gpu, b_gpu, t2)
        # @cuda threads=288 blocks=cld(length(t2), 288) gpu_adv!(adv, Ax, Ay, χx_gpu, χy_gpu, b_gpu, t2)
    end

    # copy result to CPU
    cpu_adv = Array(adv)
    return cpu_adv
end

function evolve!(m::ModelSetup3D, s::ModelState3D, t_final, t_save; Δt, i_save=0)
    # unpack
    ε² = m.params.ε²
    μϱ = m.params.μϱ
    g1 = m.geom.g1
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    g_sfc2 = m.geom.g_sfc2
    HM = m.evolution.HM
    K_hdiff = m.evolution.K_hdiff
    advection_on = m.evolution.advection

    if advection_on
        # stiffness matrix for stabilizing diffusion
        κ_h = 1e-2*ε²/μϱ
        @printf("κ_h = %1.1e\n", κ_h)
        # LHS_hdiff = CuSparseMatrixCSC(HM + κ_h*Δt/4*K_hdiff) # Δt = Δt/2
        # RHS_hdiff = CuSparseMatrixCSC(HM - κ_h*Δt/4*K_hdiff)
        LHS_hdiff = HM + κ_h*Δt/4*K_hdiff # Δt = Δt/2
        RHS_hdiff = HM - κ_h*Δt/4*K_hdiff
        # e_idxs = unique(vcat(g2.e["sfc"], g2.e["coast"]))
        # LHS_hdiff[e_idxs, :] .= 0
        # RHS_hdiff[e_idxs, :] .= 0
        # LHS_hdiff[[CartesianIndex(i, i) for i ∈ e_idxs]] .= 1
        Pinv_hdiff = sparse(inv(Diagonal(LHS_hdiff)))

        # put on GPU
        HM_gpu = CuSparseMatrixCSC(HM)
        Pinv_adv = CuSparseMatrixCSC(sparse(inv(Diagonal(HM))))
        adv = CUDA.zeros(eltype(HM_gpu), g2.np) # pre-allocate for `cg!`
        LHS_hdiff = CuSparseMatrixCSC(LHS_hdiff)
        RHS_hdiff = CuSparseMatrixCSC(RHS_hdiff)
        Pinv_hdiff = CuSparseMatrixCSC(Pinv_hdiff)
        CUDA.memory_status()
    end


    # stiffness matrix for vertical diffusion
    LHS_diffs, RHS_diffs = build_diffusion_matrices(m, Δt)

    # element map TODO: add this to `Grid`?
    el_map = build_element_map(g2)

    # timestep
    t_current = s.t[1]
    n_steps = Int64(round((t_final - t_current)/Δt))
    n_steps_save = Int64(round(t_save/Δt))

    # initial condition
    ∫b₀ = sum(HM*s.b.values)
    pe₀ = potential_energy(m, s)
    println("\nInitial condition:") 
    @printf("    ∫b₀ = % .5e\n    pe₀ = % .5e\n", ∫b₀, pe₀)
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
        # stabilizing diffusion
        if advection_on
            b_gpu = CuArray(s.b.values)
            cg!(b_gpu, LHS_hdiff, RHS_hdiff*b_gpu, Pinv=Pinv_hdiff)
            s.b.values[:] = Array(b_gpu)
        end

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

            # advect first half-step
            adv_el = advection(m, s.χx.values, s.χy.values, s.b.values)
            adv_node_gpu = CuArray(el_map*adv_el[:])
            cg!(adv, HM_gpu, -adv_node_gpu, Pinv=Pinv_adv)

            # advect second half-step
            adv_el = advection(m, s.χx.values, s.χy.values, s.b.values .+ Δt/2*Array(adv))
            adv_node_gpu = CuArray(el_map*adv_el[:])
            cg!(adv, HM_gpu, -adv_node_gpu, Pinv=Pinv_adv)

            # update
            s.b.values[:] = s.b.values + Δt*Array(adv)
        end
        # end

        # Δt/2 diffusion step
        # @time "vdiff" begin
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end
        # end

        # stabilizing diffusion
        if advection_on
            b_gpu = CuArray(s.b.values)
            cg!(b_gpu, LHS_hdiff, RHS_hdiff*b_gpu, Pinv=Pinv_hdiff)
            s.b.values[:] = Array(b_gpu)
        end

        # set time
        s.t[1] = s.t[1] + Δt

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end

        if mod(i, n_steps_save) == 0 || i == n_steps
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
            # ba = [ba_adv(g2.p[j, :], i*Δt, H[get_i_sfc(j, nσ)]) for j=1:g2.np]
            # @printf "Max Err = %1.1e\n" maximum(abs.(s.b.values - ba))
            # # diffusion solution
            # ba = [ba_diff(g2.p[j, 3], i*Δt, α/(1-g2.p[j, 1]^2-g2.p[j, 2]^2)^2, 1-g2.p[j, 1]^2-g2.p[j, 2]^2) for j=1:g2.np]

            # debug plot
            if !m.evolution.advection
                invert!(m, s)
            end
            quick_plot(s.Ψ, cb_label=L"Barotropic streamfunction $\Psi$", title=latexstring(L"$t = $", @sprintf("%1.1e", s.t[1])), filename=@sprintf("%s/images/psi%03d.png", out_folder, i_save))
            plot_u(m, s, 0; i=i_save)
            plot_profiles(m, s, x=0.5, y=0.0, filename=@sprintf("%s/images/profiles%03d.png", out_folder, i_save))

            # HDF5
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
    # return exp(-((x[1] - t)^2 + x[2]^2 + (H*x[3] + 0.5)^2)/0.02)
    # return exp(-(x[1]^2 + (x[2] - t)^2 + (H*x[3] + 0.5)^2)/0.02)
    return exp(-(x[1]^2 + x[2]^2 + (H*x[3] + 0.5 - t)^2)/0.02)
end