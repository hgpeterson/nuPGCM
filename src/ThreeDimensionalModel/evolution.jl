struct EvolutionComponents{HM, FAV, MV, A}
    # depth-weighted mass matrix
    HM::HM # sparse matrix

    K_stab::HM

    # vectors of diffusion LHS and RHS matrices for each 1D σ-column
    LHS_diffs::FAV # vector of factorized matrices
    RHS_diffs::MV # vector of sparse matrices

    # advection arrays
    Ax::A # 4D array
    Ay::A

    # advection on or off?
    advection::Bool
end

"""
    evolution = EvolutionComponents(params::Params, geom::Geometry, forcing::Forcing, advection)
"""
function EvolutionComponents(params::Params, geom::Geometry, forcing::Forcing, advection)
    # unpack
    g1 = geom.g1
    g2 = geom.g2
    H = geom.H
    nσ = geom.nσ

    K_stab = build_K_stab(geom)

    LHS_diffs, RHS_diffs = build_diffusion_matrices(params, geom, forcing)

    if advection
        HM = build_HM(g2, H, nσ)
        Ax, Ay = build_advection_arrays(g1, g2)
    else
        HM = spzeros(g2.np, g2.np)
        Ax = Ay = zeros(1, 1, 1, 1)
    end

    return EvolutionComponents(HM, K_stab, LHS_diffs, RHS_diffs, Ax, Ay, advection)
end

function build_K_stab(geom::Geometry)
    # unpack
    g = geom.g2
    el = g.el
    nσ = geom.nσ
    H = geom.H
    Hx = geom.Hx
    Hy = geom.Hy

    # σ FE
    σ = FEField(g.p[:, 3], g)

    # integrand function
    function f(ξ, i, j, k, k_sfc, jac)
        ∇φi_ref = [∂φ(el, ξ, i, d) for d=1:3]
        ∇φj_ref = [∂φ(el, ξ, j, d) for d=1:3]
        ∇φi = jac'*∇φi_ref
        ∇φj = jac'*∇φj_ref
        return g.J.dets[k]*(∇φi[1]*(∇φj[1]*H(ξ[1:2], k_sfc) - σ(ξ, k)*Hx(ξ[1:2], k_sfc)*∇φj[3]) + 
                            ∇φi[2]*(∇φj[2]*H(ξ[1:2], k_sfc) - σ(ξ, k)*Hy(ξ[1:2], k_sfc)*∇φj[3]) + 
                            ∇φi[3]*((1 + σ(ξ, k)^2*(Hx(ξ[1:2], k_sfc)^2 + Hy(ξ[1:2], k_sfc)^2))/H(ξ[1:2], k_sfc)*∇φj[3] - 
                            σ(ξ, k)*Hx(ξ[1:2], k_sfc)*∇φj[1] - 
                            σ(ξ, k)*Hy(ξ[1:2], k_sfc)*∇φj[2]))
    end

    # stamp
    N = g.nt*el.n^2
    I = zeros(Int64, N)
    J = zeros(Int64, N)
    V = zeros(Float64, N)
    n = 1
    @showprogress "Building diffusion matrix..." for k=1:g.nt
        k_sfc = get_k_sfc(k, nσ)
        jac = g.J.Js[k, :, :]
        for i=1:el.n, j=1:el.n
            I[n] = g.t[k, i]
            J[n] = g.t[k, j]
            V[n] = ref_el_quad(ξ -> f(ξ, i, j, k, k_sfc, jac), el)
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
    # ∂σ(b) = 0 at z = 0
    fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
    push!(K, (nσ, nσ-2, fd_σ[1]))
    push!(K, (nσ, nσ-1, fd_σ[2]))
    push!(K, (nσ, nσ  , fd_σ[3]))
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), nσ, nσ))
end

"""
    LHSs, RHSs = build_diffusion_matrices(params::Params, geom::Geometry, forcing::Forcing)
"""
function build_diffusion_matrices(params::Params, geom::Geometry, forcing::Forcing)
    # unpack
    ε² = params.ε²
    μ = params.μ
    ϱ = params.ϱ
    Δt = params.Δt
    σ = geom.σ
    nσ = geom.nσ
    H = geom.H
    κ = forcing.κ
    g_sfc2 = geom.g_sfc2

    α = ε²/μ/ϱ
    K_cols = [build_K_col(σ, κ[get_col_inds(i, geom.nσ)]) for i=1:g_sfc2.np]
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

function build_element_map(g)
    imap = reshape(1:length(g.t), g.nt, g.nn)
    A = Tuple{Int64,Int64,Int64}[]
    for i=1:g.np
        for I ∈ g.p_to_t[i]
            push!(A, (i, imap[I], 1))
        end
    end
    return dropzeros!(sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), g.np, length(g.t)))
end

function evolve!(m::ModelSetup3D, s::ModelState3D, t_final, t_plot, t_save)
    # unpack
    Δt = m.params.Δt
    g1 = m.geom.g1
    g2 = m.geom.g2
    nσ = m.geom.nσ
    H = m.geom.H
    g_sfc2 = m.geom.g_sfc2
    LHS_diffs = m.evolution.LHS_diffs
    RHS_diffs = m.evolution.RHS_diffs
    HM = m.evolution.HM
    K_stab = m.evolution.K_stab

    # put on GPU
    HM_gpu = CuSparseMatrixCSC(HM)

    # stiffness matrix for stabilizing diffusion
    # LHS_stab = CuSparseMatrixCSC(HM + 2e-5*Δt/2*K_stab)
    LHS_stab = CuSparseMatrixCSC(HM + 1e-4*Δt/2*K_stab)

    # element map TODO: add this to `Grid`?
    el_map = build_element_map(g2)

    # timestep
    n_steps = Int64(round(t_final/Δt))
    n_steps_plot = Int64(round(t_plot/Δt))
    n_steps_save = Int64(round(t_save/Δt))

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
    ∫b₀ = sum(HM*s.b.values)
    pe₀ = potential_energy(m, s)
    @info "Initial condition" ∫b₀ pe₀
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
    # adv = zeros(g2.np) # pre-allocate for `cg!`
    t0 = time()
    t1 = time()
    i_save = 1
    for i=1:n_steps
        s.b.values[:] = Array(cg(LHS_stab, CuArray(HM*s.b.values)))

        # @time "diffusion" begin
        # Δt/2 vertical diffusion step
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end
        # end

        # @time "advection" begin
        # Δt advection step
        if m.evolution.advection
            invert!(m, s)
            adv_el = advection(m, s.χx.values, s.χy.values, s.b.values)
            adv_node_gpu = CuArray(el_map*adv_el[:])
            adv = Array(cg(HM_gpu, -adv_node_gpu))
            adv_el = advection(m, s.χx.values, s.χy.values, s.b.values .+ Δt/2*adv)
            adv_node_gpu = CuArray(el_map*adv_el[:])
            adv = Array(cg(HM_gpu, -adv_node_gpu))
            s.b.values[:] = s.b.values + Δt*adv
        end
        # end

        # @time "diffusion" begin
        # Δt/2 diffusion step
        for j=1:g_sfc2.np
            inds = get_col_inds(j, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end
        # end

        s.b.values[:] = Array(cg(LHS_stab, CuArray(HM*s.b.values)))

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end

        if mod(i, n_steps_save) == 0 || i == n_steps
            # info
            t_elapsed0 = time() - t0
            t_elapsed1 = time() - t1
            t1 = time()
            ∫b = sum(HM*s.b.values) 
            Δb = abs(∫b - ∫b₀) 
            Δb_pct = 100*abs(Δb/∫b₀)
            # ETR = (n_steps - i)*t_elapsed1/n_steps_save
            ETR = (n_steps - i)*t_elapsed0/i
            ETR_h = ETR ÷ 3600
            ETR_m = (ETR % 3600) ÷ 60
            ETR_s = ETR % 60
            @info @sprintf("%d steps in %d s (ETR: %02d:%02d:%02d)", i, t_elapsed0, ETR_h, ETR_m, ETR_s) ∫b Δb Δb_pct
            ux = [-∂z(s.χy, [0, 0, 0], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            uy = [+∂z(s.χx, [0, 0, 0], k)/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            uσ = [(∂x(s.χy, [0, 0, 0], k) - ∂y(s.χx, [0, 0, 0], k))/sum(H[g_sfc2.t[get_k_sfc(k, nσ), :]]/6) for k=1:g1.nt]
            @info "CFL" Δt_x=minimum(dx./abs.(ux)) Δt_y=minimum(dy./abs.(uy)) Δt_σ=minimum(dσ./abs.(uσ)) Δt

            # # energy 
            # b_prod = buoyancy_production(m, s) 
            # ke_diss = KE_dissipation(m, s)
            # println("KE:")
            # @printf("    b_prod  = %1.1e\n", b_prod)
            # @printf("    ke_diss = %1.1e\n", ke_diss)
            # @printf("    ke_loss = %1.1e\n", abs(ke_diss-b_prod))
            # pe = potential_energy(m, s)
            # pe_prod = PE_production(m, s)
            # println("PE:")
            # @printf("    pe      = %1.1e\n", pe)
            # @printf("    Δpe     = %1.1e\n", pe - pe₀)
            # @printf("    pe_prod = %1.1e\n", pe_prod)

            # # advection solution
            # ba = [ba_adv(g2.p[j, :], i*Δt, H[get_i_sfc(j, nσ)]) for j=1:g2.np]
            # @printf "Max Err = %1.1e\n" maximum(abs.(s.b.values - ba))
            # # diffusion solution
            # ba = [ba_diff(g2.p[j, 3], i*Δt, α/(1-g2.p[j, 1]^2-g2.p[j, 2]^2)^2, 1-g2.p[j, 1]^2-g2.p[j, 2]^2) for j=1:g2.np]

            # # save state 
            # cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)] 
            # vtk_grid("$out_folder/state$i", pz', cells) do vtk 
            #     vtk["b"] = s.b.values[1:g1.np] 
            #     # vtk["ba"] = ba[1:g1.np] 
            #     # vtk["err"] = abs.(s.b.values[1:g1.np] - ba[1:g1.np]) 
            #     vtk["ωx"] = FEField(s.ωx).values 
            #     vtk["ωy"] = FEField(s.ωy).values 
            #     vtk["χx"] = FEField(s.χx).values 
            #     vtk["χy"] = FEField(s.χy).values 
            #     pvd[i*Δt] = vtk 
            # end 
            # println("$out_folder/state$i.vtu") 
            # cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)] 
            # vtk_grid("$out_folder/stateTF$i", g1.p', cells) do vtk 
            #     vtk["b"] = s.b.values[1:g1.np] 
            # end 
            # println("$out_folder/stateTF$i.vtu") 

            # HDF5
            save_state(s, "$out_folder/state$i_save.h5")
            i_save += 1

            flush(stdout)
            flush(stderr)
        end

        if mod(i, n_steps_plot) == 0 || i == n_steps
            # show state
            invert!(m, s, showplots=true)
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
    # return exp(-((x[1] - t)^2 + x[2]^2 + (H*x[3] + 0.5)^2)/0.02)
    # return exp(-(x[1]^2 + (x[2] - t)^2 + (H*x[3] + 0.5)^2)/0.02)
    return exp(-(x[1]^2 + x[2]^2 + (H*x[3] + 0.5 - t)^2)/0.02)
end