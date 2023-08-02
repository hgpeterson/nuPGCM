function get_K_col(g, κ)
    κ = FEField(κ, g)
    J = g.J
    el = g.el
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt
        # ∫ ν ∂φᵢ∂φⱼ
        σ(ξ) = transform_from_ref_el(el, ξ, g.p[g.t[k, :]])
        κK = [ref_el_quad(ξ -> κ(σ(ξ), k)*φξ(el, ξ, i)*φξ(el, ξ, j)*J.Js[k, 1, 1]^2*J.dets[k], el) for i=1:el.n, j=1:el.n]
        for i=1:el.n, j=1:el.n
            push!(K, (g.t[k, i], g.t[k, j], κK[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np))
end

"""
    HM = get_HM(g2, H, nσ)

Compute `HM` = ∫ `H` φᵢ φⱼ for second order 3D grid `g2` with `nσ` vertical nodes.
"""
function get_HM(g2, H::FEField, nσ)
    # unpack
    g_sfc2 = H.g
    tri2 = g_sfc2.el
    J = g2.J
    w2 = g2.el

    # stamp
    HM = Tuple{Int64, Int64, Float64}[]
    @showprogress "Building depth-weighted mass matrix..." for k_sfc=1:g_sfc2.nt
        x(ξ) = transform_from_ref_el(tri2, ξ[1:2], g_sfc2.p[g_sfc2.t[k_sfc, 1:3], :])
        f(ξ, i, j) = H(x(ξ), k_sfc)*φ(w2, ξ, i)*φ(w2, ξ, j)
        HM_el = [ref_el_quad(ξ -> f(ξ, i, j), w2) for i=1:w2.n, j=1:w2.n]
        for k_w=get_k_ws(k_sfc, nσ), i=1:w2.n, j=1:w2.n
            push!(HM, (g2.t[k_w, i], g2.t[k_w, j], J.dets[k_w]*HM_el[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(HM), (x->x[2]).(HM), (x->x[3]).(HM), g2.np, g2.np))
end

"""
    Aξ, Aη, Aσξ, Aση = get_advection_arrays(g1, g2)

Compute advection arrays of the form ∫ φᵢ∂φⱼ∂φₖ where φᵢ and φⱼ are defined on the 
second order grid `g2` and φₖ is defined on the first order grid `g1`. These are then
multiplied by the proper Jacobian terms to get the arrays:
    • `Aξ` for the -∂σ(χy)*∂ξ(b) term,
    • `Aη` for the ∂σ(χx)*∂η(b) term, and
    • `Aσξ` and `Aση` for the [∂ξ(χy) - ∂η(χx)]*∂σ(b) term.
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
    Aξ  = zeros(g1.nt, w2.n, w2.n, w1.n)
    Aη  = zeros(g1.nt, w2.n, w2.n, w1.n)
    Aσξ = zeros(g1.nt, w2.n, w2.n, w1.n)
    Aση = zeros(g1.nt, w2.n, w2.n, w1.n)

    @showprogress "Setting up advection arrays..." for k=1:g1.nt
        # unpack
        jac = J.Js[k, :, :]
        Δ = J.dets[k]

        # -∂σ(χy)*∂ξ(b)
        Aξ[k, :, :, :] = -sum(A[:, :, :, d1, d2]*jac[d1, 3]*jac[d2, 1]*Δ for d1=1:3, d2=1:3)

        # ∂σ(χx)*∂η(b)
        Aη[k, :, :, :] = sum(A[:, :, :, d1, d2]*jac[d1, 3]*jac[d2, 2]*Δ for d1=1:3, d2=1:3)

        # [∂ξ(χy) - ∂η(χx)]*∂σ(b)
        Aσξ[k, :, :, :] = sum(A[:, :, :, d1, d2]*jac[d1, 1]*jac[d2, 3]*Δ for d1=1:3, d2=1:3)
        Aση[k, :, :, :] = -sum(A[:, :, :, d1, d2]*jac[d1, 2]*jac[d2, 3]*Δ for d1=1:3, d2=1:3)
    end

    return Aξ, Aη, Aσξ, Aση
end

function advection(m::ModelSetup3D, χx, χy, b)
    g1 = m.g1
    g2 = m.g2
    adv = zeros(g2.np)
    for k=1:g2.nt, i=1:g2.nn
        adv[g2.t[k, i]] += sum(m.Aξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χy[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(m.Aη[k, i, ib, iχ]*b[g2.t[k, ib]]*χx[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(m.Aσξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χy[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(m.Aση[k, i, ib, iχ]*b[g2.t[k, ib]]*χx[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn)
    end
    return adv
end

# function RK2(f, u, Δt)
#     return u + Δt*f(u + Δt/2*f(u))
# end

function evolve!(m::ModelSetup3D, s::ModelState3D)
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
    g_col = m.g_col
    in_nodes2 = m.in_nodes2

    # integration time
    α = ε²/μ/ϱ
    T = 5e-2/α
    n_steps = 50
    Δt = T/n_steps
    # Δt = 1e-3
    # n_steps = 11

    # # constant vel. (ux = 1, uy = 0, uz = 0, or uξ = 1, uη = 0, uσ = -σHₓ/H)
    # s.χx.values[:] .= 0.0
    # # s.χy.values[:] = @. -g1.p[g1.t, 3]*(1 - g1.p[g1.t, 1]^2 - g1.p[g1.t, 2]^2)^3
    # s.χy.values[:] = @. -g1.p[g1.t, 3]*(2 - g1.p[g1.t, 1]^2 - g1.p[g1.t, 2]^2)
    # println(@sprintf("CFL Δt: %1.1e", min(1/sqrt(g_sfc1.np), 1/2/(nσ - 1))))
    # println(@sprintf("    Δt: %1.1e", Δt))

    # diffusion matrices
    M_col = mass_matrix(g_col)
    K_cols = [get_K_col(g_col, κ[get_col_inds(i, nσ)]) for i=1:g_sfc2.np]
    LHS_diffs = [lu(M_col + α/H[i]^2*Δt/2*K_cols[i]) for i ∈ in_nodes2]
    RHS_diffs = [M_col - α/H[i]^2*Δt/2*K_cols[i] for i ∈ in_nodes2]

    # pvd file
    rm("$out_folder/state.pvd", force=true)
    # rm("$out_folder/state*.vtu", force=true) # * doesn't work?
    pvd = paraview_collection("$out_folder/state", append=true)

    # for plotting
    pz = copy(g1.p)
    for i=1:g1.np
        pz[i, 3] *= 2 - pz[i, 1]^2 - pz[i, 2]^2
    end

    # initial condition
    println(@sprintf("∫b₀ = %1.5e", sum(HM*s.b.values)))
    cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
    vtk_grid("$out_folder/state0", pz', cells) do vtk
        vtk["b"] = s.b.values[1:g1.np]
        vtk["ba"] = s.b.values[1:g1.np]
        vtk["err"] = zeros(g1.np)
        pvd[0] = vtk
    end
    println("$out_folder/state0.vtu")

    # solve
    adv = zeros(g2.np) # pre-allocate for cg!
    t0 = time()
    for i=1:n_steps
        # # rk2
        # cg!(adv, HM, advection(m, s.χx, s.χy, s.b))
        # bhalf = s.b.values - Δt/2*adv
        # cg!(adv, HM, advection(m, s.χx, s.χy, bhalf))
        # s.b.values[:] = s.b.values - Δt*adv

        # # Δt/2 advection step
        # invert!(m, s)
        # cg!(adv, HM, advection(m, s.χx, s.χy, s.b))
        # s.b.values[:] = s.b.values - Δt/2*adv

        # Δt diffusion step
        for j ∈ eachindex(in_nodes2)
            ig = in_nodes2[j]
            inds = get_col_inds(ig, nσ)
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end

        # # Δt/2 advection step
        # invert!(m, s)
        # cg!(adv, HM, advection(m, s.χx, s.χy, s.b))
        # s.b.values[:] = s.b.values - Δt/2*adv

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end

        if mod(i, 10) == 0 || i == n_steps
            # # advection solution
            # ba = [ba_adv(g2.p[j, :], i*Δt) for j=1:g2.np]
            # # diffusion solution
            # ba = [ba_diff(g2.p[j, 3], i*Δt, α/(1-g2.p[j, 1]^2-g2.p[j, 2]^2)^2, 1-g2.p[j, 1]^2-g2.p[j, 2]^2) for j=1:g2.np]

            # info
            println(@sprintf("\n%d steps in %d s", i, time()-t0))
            # println(@sprintf("Max Error: %1.1e", maximum(abs.(s.b.values - ba))))
            println(@sprintf("∫b = %1.5e", sum(HM*s.b.values)))

            # show state
            invert!(m, s, showplots=true)

            # save state
            cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
            vtk_grid("$out_folder/state$i", pz', cells) do vtk
                vtk["b"] = s.b.values[1:g1.np]
                # vtk["ba"] = ba[1:g1.np]
                # vtk["err"] = abs.(s.b.values[1:g1.np] - ba[1:g1.np])
                vtk["ωξ"] = FEField(s.ωx).values
                vtk["ωη"] = FEField(s.ωy).values
                vtk["χξ"] = FEField(s.χx).values
                vtk["χη"] = FEField(s.χy).values
                pvd[i*Δt] = vtk
            end
            println("$out_folder/state$i.vtu")
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

    # # b0 = H^3*(σ^2 + 2/3*σ^3), nuemann
    # A(n) = 8*H^3*(-1 + (-1)^n)/(n^4*π^4)
    # return H^3/6 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
end

"""
    ba = ba_adv(x, t)

Analytical solution to ∂t(b) + ∂x(b) = 0 for gaussian initial condition.
"""
function ba_adv(x, t)
    return exp(-((x[1] - t)^2 + x[2]^2 + ((2 - x[1]^2 - x[2]^2)*x[3] + 0.5)^2)/0.02)
end

## advection convergence tests with H = 2 - x^2 - y^2

# Δt = 1e-3, n_steps = 11
# mesh  error
# 0     6.7e-3
# 1     2.7e-3
# 2     4.7e-3
# 3     1.7e-3

# mesh 2, T = 1e-2
# nsteps  error
# 2       2.4e-3
# 16      4.4e-3
# 128     4.6e-3

# mesh 3, T = 1e-2
# nsteps  error
# 2       8.8e-4 
# 16      1.6e-3